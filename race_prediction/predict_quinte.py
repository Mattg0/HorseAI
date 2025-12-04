#!/usr/bin/env python3
"""
Quinté Race Prediction Script

Loads quinté races from daily_race table, applies trained quinté models,
and saves predictions to CSV/JSON files.

Usage:
    python race_prediction/predict_quinte.py [--date YYYY-MM-DD] [--output-dir path]
"""

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np

from utils.env_setup import AppConfig, get_sqlite_dbpath
from utils.model_manager import ModelManager
from core.calculators.static_feature_calculator import FeatureCalculator
from core.calculators.quinte_feature_calculator import QuinteFeatureCalculator
from race_prediction.competitive_field_analyzer import CompetitiveFieldAnalyzer
from model_training.regressions.adaptive_calibrator import AdaptiveCalibratorManager


class QuintePredictionEngine:
    """
    Prediction engine for Quinté+ races.

    Loads quinté-specific models and applies them to daily quinté races.
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = True,
                 quinte_weight: float = 0.20, general_weight: float = 0.80,
                 use_general_blend: bool = True):
        """
        Initialize the quinté prediction engine.

        Args:
            config_path: Path to config file
            verbose: Whether to print verbose output
            quinte_weight: Weight for quinté model (default: 0.20 based on optimization)
            general_weight: Weight for general model (default: 0.80 based on optimization)
            use_general_blend: Whether to blend with general model predictions
        """
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Blend weights (optimized values from blend optimization)
        self.quinte_weight = quinte_weight
        self.general_weight = general_weight
        self.use_general_blend = use_general_blend

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        self.db_path = get_sqlite_dbpath(self.db_type)

        # Load quinté models
        self.model_manager = ModelManager()
        self.log_info("Loading quinté models...")

        self.rf_model = None
        self.tabnet_model = None
        self.scaler = None
        self.feature_columns = None
        self.feature_selector = None  # For TabNet feature selection
        self.tabnet_model_path = None  # Store model path for loading feature selector

        self._load_models()

        # Initialize quinté feature calculator
        self.quinte_calculator = QuinteFeatureCalculator(self.db_path)

        # Initialize competitive field analyzer
        self.competitive_analyzer = CompetitiveFieldAnalyzer(verbose=self.verbose, db_path=self.db_path)

        # Initialize Quinté-specific prediction storage
        from race_prediction.quinte_prediction_storage import QuintePredictionStorage
        self.prediction_storage = QuintePredictionStorage(db_path=self.db_path, verbose=self.verbose)

        # Initialize adaptive calibrators
        self.calibrator_manager = AdaptiveCalibratorManager()
        self.rf_calibrator = None
        self.tabnet_calibrator = None
        self._load_calibrators()

        self.log_info(f"Initialized QuintePredictionEngine with database: {self.db_type}")
        if self.use_general_blend:
            self.log_info(f"  Blend weights: Quinté={self.quinte_weight:.2f}, General={self.general_weight:.2f}")
        self.log_info("✓ Competitive field analyzer initialized")
        self.log_info("✓ Prediction storage initialized")

    def log_info(self, message):
        """Simple logging method."""
        if self.verbose:
            print(f"[QuintePrediction] {message}")

    def _load_models(self):
        """Load trained quinté models."""
        try:
            # Load Random Forest quinté model (now XGBoost)
            rf_info = self.model_manager.load_quinte_model('rf')
            if rf_info and 'model' in rf_info:
                self.rf_model = rf_info['model']
                self.log_info(f"✓ Loaded RF quinté model from {rf_info['path']}")
            else:
                if rf_info:
                    self.log_info(f"⚠ RF quinté model path found but model file missing: {rf_info.get('path')}")
                else:
                    self.log_info("⚠ No RF quinté model found")

            # Load TabNet quinté model
            tabnet_info = self.model_manager.load_quinte_model('tabnet')
            if tabnet_info and 'model' in tabnet_info:
                self.tabnet_model = tabnet_info['model']
                self.scaler = tabnet_info.get('scaler')
                self.feature_columns = tabnet_info.get('feature_columns', [])
                self.tabnet_model_path = Path(tabnet_info['path']).parent  # Store model directory
                self.log_info(f"✓ Loaded TabNet quinté model from {tabnet_info['path']}")
                self.log_info(f"  Features: {len(self.feature_columns)}")

                # Try to load feature selector if it exists
                try:
                    from core.feature_selection.tabnet_feature_selector import TabNetFeatureSelector
                    feature_selector_path = self.tabnet_model_path / "feature_selector.json"
                    if feature_selector_path.exists():
                        self.feature_selector = TabNetFeatureSelector.load(str(feature_selector_path))
                        self.log_info(f"✓ Loaded feature selector: {len(self.feature_selector.selected_features)} selected features")
                    else:
                        self.log_info(f"  No feature selector found (using all {len(self.feature_columns)} features)")
                except Exception as e:
                    self.log_info(f"  Warning: Could not load feature selector: {e}")

                # FAIL-FAST VALIDATION: Check if scaler expects the same features
                if self.scaler is not None and hasattr(self.scaler, 'feature_names_in_'):
                    scaler_features = set(self.scaler.feature_names_in_)
                    model_features = set(self.feature_columns)

                    if scaler_features != model_features:
                        missing_in_scaler = model_features - scaler_features
                        extra_in_scaler = scaler_features - model_features

                        error_msg = f"""
❌ FEATURE MISMATCH: Model and Scaler are misaligned!

Model expects: {len(model_features)} features
Scaler expects: {len(scaler_features)} features

Missing in scaler (model has these): {len(missing_in_scaler)}
  {list(missing_in_scaler)[:10]}

Extra in scaler (model doesn't have): {len(extra_in_scaler)}
  {list(extra_in_scaler)[:10]}

🔧 FIX: You need to RETRAIN the model with the updated training script.
   Run: python model_training/historical/train_quinte_model.py

This will generate a new model + scaler with aligned features.
"""
                        self.log_info(error_msg)
                        raise ValueError("Model/Scaler feature mismatch. Retrain required.")

            else:
                if tabnet_info:
                    self.log_info(f"⚠ TabNet quinté model path found but model file missing: {tabnet_info.get('path')}")
                else:
                    self.log_info("⚠ No TabNet quinté model found")

            if not self.rf_model and not self.tabnet_model:
                raise ValueError("No quinté models found. Please train models first.")

        except Exception as e:
            self.log_info(f"Error loading models: {e}")
            raise

    def _load_calibrators(self):
        """Load adaptive calibrators for RF and TabNet models."""
        self.log_info("Loading adaptive calibrators...")

        # Load RF calibrator
        self.rf_calibrator = self.calibrator_manager.load_calibrator('rf')
        if self.rf_calibrator:
            metadata = self.calibrator_manager.load_metadata('rf')
            if metadata:
                self.log_info(f"✓ Loaded RF calibrator:")
                self.log_info(f"   Data points: {metadata['data_points']}")
                self.log_info(f"   MAE improvement: {metadata['mae_improvement_pct']:.2f}%")
                self.log_info(f"   Last updated: {metadata['last_updated']}")
        else:
            self.log_info("  No RF calibrator found (will use raw predictions)")

        # Load TabNet calibrator
        self.tabnet_calibrator = self.calibrator_manager.load_calibrator('tabnet')
        if self.tabnet_calibrator:
            metadata = self.calibrator_manager.load_metadata('tabnet')
            if metadata:
                self.log_info(f"✓ Loaded TabNet calibrator:")
                self.log_info(f"   Data points: {metadata['data_points']}")
                self.log_info(f"   MAE improvement: {metadata['mae_improvement_pct']:.2f}%")
                self.log_info(f"   Last updated: {metadata['last_updated']}")
        else:
            self.log_info("  No TabNet calibrator found (will use raw predictions)")

    def load_daily_quinte_races(self, race_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load quinté races from daily_race table.

        Args:
            race_date: Specific date to load (YYYY-MM-DD format), or None for all quinté races

        Returns:
            DataFrame with quinté race data
        """
        conn = sqlite3.connect(self.db_path)

        if race_date is None:
            # Load ALL quinté races from daily_race table
            self.log_info(f"Loading ALL quinté races from daily_race...")
            query = """
            SELECT * FROM daily_race
            WHERE quinte = 1
            ORDER BY jour DESC, reun, prix
            """
            df_races = pd.read_sql_query(query, conn)
        else:
            # Load quinté races for specific date
            self.log_info(f"Loading quinté races for {race_date}...")
            query = """
            SELECT * FROM daily_race
            WHERE quinte = 1 AND jour = ?
            ORDER BY reun, prix
            """
            df_races = pd.read_sql_query(query, conn, params=(race_date,))

        conn.close()

        self.log_info(f"Loaded {len(df_races)} quinté races")

        # Show date range if loading all races
        if race_date is None and len(df_races) > 0:
            min_date = df_races['jour'].min()
            max_date = df_races['jour'].max()
            unique_dates = df_races['jour'].nunique()
            self.log_info(f"  Date range: {min_date} to {max_date} ({unique_dates} days)")

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

    def prepare_features(self, df_participants: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare complete feature set for quinté prediction.

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

        # Step 2: Batch-load all quinté historical data
        self.log_info("Batch-loading quinté historical data...")
        all_quinte_data = self.quinte_calculator.batch_load_all_quinte_data()

        self.log_info(f"Loaded {len(all_quinte_data['races'])} historical races")

        # Step 3: Add quinté features using batch-loaded data
        self.log_info("Calculating quinté features...")

        unique_races = df_with_features['comp'].unique()

        # Process each race separately to add quinté features
        race_dfs = []
        for race_comp in unique_races:
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
                    all_data=all_quinte_data
                )
                race_dfs.append(race_df)
            except Exception as e:
                self.log_info(f"Warning: Could not add quinté features for race {race_comp}: {e}")
                race_dfs.append(race_df)

        # Combine all races
        df_complete = pd.concat(race_dfs, ignore_index=True)

        self.log_info(f"Complete feature set: {len(df_complete)} participants, {len(df_complete.columns)} features")

        return df_complete

    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using trained quinté models.

        Args:
            df_features: DataFrame with all features

        Returns:
            DataFrame with predictions added
        """
        self.log_info("Generating predictions...")

        result_df = df_features.copy()

        # Prepare feature matrix (MUST match training exactly - NO cleanup, only transformations)
        from core.data_cleaning.feature_cleanup import FeatureCleaner
        cleaner = FeatureCleaner()

        # Map transformed feature names back to original names
        transform_mapping = {
            'recence_log': 'recence',
            'cotedirect_log': 'cotedirect'
        }

        # If feature selector exists, we need to apply it to the FULL feature set
        # The selector knows which features to keep
        if self.feature_selector is not None:
            self.log_info(f"Using TabNet feature selection pipeline...")

            # Get the ORIGINAL features that were used during training (before selection)
            # These are saved in the feature_selector metadata
            original_training_features = self.feature_selector.original_features

            # Select ALL original training features from result_df
            available_features = []
            missing_features = []
            for col in original_training_features:
                original_name = transform_mapping.get(col, col)
                if original_name in result_df.columns:
                    available_features.append(original_name)
                else:
                    missing_features.append(col)

            if missing_features and self.verbose:
                self.log_info(f"⚠️  {len(missing_features)}/{len(original_training_features)} features missing")

            X_full = result_df[available_features].copy()

            # Apply transformations (recence→recence_log, cotedirect→cotedirect_log)
            X_full = cleaner.apply_transformations(X_full)
            X_full = X_full.fillna(0)

            # Remove DNF_rate features (consistency with training)
            dnf_features = [col for col in X_full.columns if 'dnf_rate' in col.lower()]
            if dnf_features:
                X_full = X_full.drop(columns=dnf_features)

            # Save X_full for RF model BEFORE applying feature selection
            X_rf = X_full.copy()  # RF uses ALL original features

            # Apply feature selection for TabNet
            X_tabnet = self.feature_selector.transform(X_full)

            if self.verbose:
                self.log_info(f"Feature selection: {len(X_full.columns)} → {len(X_tabnet.columns)} features")

            # For validation, X should be the TabNet features (what model expects)
            X = X_tabnet

        else:
            # No feature selection - use model's expected features
            self.log_info(f"No feature selection - using model's expected features...")

            # Select features that model expects
            available_features = []
            missing_features = []
            for col in self.feature_columns:
                original_name = transform_mapping.get(col, col)
                if original_name in result_df.columns:
                    available_features.append(original_name)
                else:
                    missing_features.append(col)

            if missing_features and self.verbose:
                self.log_info(f"⚠️  {len(missing_features)}/{len(self.feature_columns)} features missing")

            X = result_df[available_features].copy()

            # Apply transformations (recence→recence_log, cotedirect→cotedirect_log)
            X = cleaner.apply_transformations(X)
            X = X.fillna(0)

            # Remove DNF_rate features (consistency with training)
            dnf_features = [col for col in X.columns if 'dnf_rate' in col.lower()]
            if dnf_features:
                X = X.drop(columns=dnf_features)

            # Both models use the same features when no feature selection
            X_rf = X.copy()
            X_tabnet = X.copy()

        # Validate feature alignment for TabNet (X_tabnet should match trained model expectations)
        # Note: X_rf may have different features if feature selection was used
        if self.feature_selector is not None:
            # With feature selection: X_tabnet should match selected features
            expected_features = self.feature_selector.selected_features
        else:
            # Without feature selection: X_tabnet should match all training features
            expected_features = self.feature_columns

        if len(X_tabnet.columns) != len(expected_features):
            missing_features = set(expected_features) - set(X_tabnet.columns)
            extra_features = set(X_tabnet.columns) - set(expected_features)

            mismatch_msg = f"""
❌ FEATURE COUNT MISMATCH!

Created: {len(X_tabnet.columns)} features
Expected: {len(expected_features)} features
Missing: {len(missing_features)} features
Extra: {len(extra_features)} features

Using feature selection: {self.feature_selector is not None}
"""
            self.log_info(mismatch_msg)
            if self.verbose:
                self.log_info(f"Missing: {sorted(list(missing_features))[:10]}")
                self.log_info(f"Extra: {sorted(list(extra_features))[:10]}")
            raise ValueError(f"Feature count mismatch: {len(X_tabnet.columns)} != {len(expected_features)}")

        # Validate feature names match (order doesn't matter)
        X_feature_set = set(X_tabnet.columns)
        expected_feature_set = set(expected_features)
        if X_feature_set != expected_feature_set:
            missing = expected_feature_set - X_feature_set
            extra = X_feature_set - expected_feature_set
            mismatch_msg = f"""
❌ FEATURE NAME MISMATCH!

Missing from prediction: {len(missing)} features
Extra in prediction: {len(extra)} features
"""
            self.log_info(mismatch_msg)
            if self.verbose:
                self.log_info(f"Missing: {sorted(list(missing))[:10]}")
                self.log_info(f"Extra: {sorted(list(extra))[:10]}")
            raise ValueError("Feature names don't match model expectations")

        # X_rf and X_tabnet are already created above (lines 404, 408, 441-442)
        # X_rf = full features for RF (92 features)
        # X_tabnet = selected features for TabNet (45 features if feature selection, else all features)

        # TabNet predictions (RAW - no isotonic calibration in model)
        if self.tabnet_model and self.scaler:
            # Ensure column order matches scaler's expected order
            if hasattr(self.scaler, 'feature_names_in_'):
                # Reorder columns to match scaler
                scaler_features = self.scaler.feature_names_in_
                # Check if features match
                if set(scaler_features) != set(X_tabnet.columns):
                    self.log_info("⚠️  Warning: Scaler features don't match X_tabnet features")
                    # Use only common features in correct order
                    common_features = [f for f in scaler_features if f in X_tabnet.columns]
                    X_tabnet = X_tabnet[common_features]
                else:
                    X_tabnet = X_tabnet[scaler_features]

            X_scaled = self.scaler.transform(X_tabnet)
            X_scaled = X_scaled.astype(np.float32)
            raw_tabnet_preds = self.tabnet_model.predict(X_scaled).flatten()

            # Store raw TabNet predictions
            result_df['raw_tabnet_prediction'] = raw_tabnet_preds

            # Apply adaptive calibration if available
            calibrated_tabnet_preds = self.calibrator_manager.apply_calibration(
                raw_tabnet_preds, self.tabnet_calibrator
            )
            result_df['predicted_position_tabnet'] = calibrated_tabnet_preds

            if self.verbose and self.tabnet_calibrator:
                self.log_info(f"  TabNet predictions calibrated")

        # Random Forest predictions (uses ORIGINAL full feature set)
        if self.rf_model:
            # Align X_rf features with what RF model expects
            expected_rf_features = None
            if hasattr(self.rf_model, 'feature_names_in_'):
                expected_rf_features = self.rf_model.feature_names_in_
            elif hasattr(self.rf_model, 'base_regressor'):
                if hasattr(self.rf_model.base_regressor, 'feature_names_in_'):
                    expected_rf_features = self.rf_model.base_regressor.feature_names_in_

            if expected_rf_features is not None:
                # Align features - reindex to match expected features
                available_rf_features = [f for f in expected_rf_features if f in X_rf.columns]
                missing_rf_features = [f for f in expected_rf_features if f not in X_rf.columns]

                if missing_rf_features:
                    if len(missing_rf_features) > 10 and self.verbose:
                        self.log_info(f"⚠️  RF missing {len(missing_rf_features)} features")
                        self.log_info(f"  First 10: {missing_rf_features[:10]}")
                    elif self.verbose:
                        self.log_info(f"⚠️  RF missing features: {missing_rf_features}")

                # Use reindex for fast alignment (fills missing with 0)
                X_rf_aligned = X_rf.reindex(columns=expected_rf_features, fill_value=0)

                # Ensure all values are numeric
                for col in X_rf_aligned.columns:
                    X_rf_aligned[col] = pd.to_numeric(X_rf_aligned[col], errors='coerce').fillna(0)

                X_rf = X_rf_aligned

            # Get RAW predictions (XGBoost predicts absolute positions directly)
            if hasattr(self.rf_model, 'predict_raw'):
                raw_rf_preds = self.rf_model.predict_raw(X_rf)
            else:
                # Fallback if not CalibratedRegressor
                raw_rf_preds = self.rf_model.predict(X_rf)

            # Store raw RF predictions
            result_df['raw_rf_prediction'] = raw_rf_preds

            # Apply adaptive calibration if available
            calibrated_rf_preds = self.calibrator_manager.apply_calibration(
                raw_rf_preds, self.rf_calibrator
            )
            result_df['predicted_position_rf'] = calibrated_rf_preds

        # Ensemble prediction (average of both models if available)
        if 'predicted_position_tabnet' in result_df.columns and 'predicted_position_rf' in result_df.columns:
            result_df['predicted_position_base'] = (
                result_df['predicted_position_tabnet'] * 0.4 +  # TabNet weight
                result_df['predicted_position_rf'] * 0.6         # RF weight
            )
        elif 'predicted_position_tabnet' in result_df.columns:
            result_df['predicted_position_base'] = result_df['predicted_position_tabnet']
        elif 'predicted_position_rf' in result_df.columns:
            result_df['predicted_position_base'] = result_df['predicted_position_rf']
        else:
            raise ValueError("No predictions generated")

        # Apply competitive field analysis to enhance predictions
        result_df = self._apply_competitive_analysis(result_df)

        # Apply two-stage refinement for positions 4-5 (based on failure analysis)
        result_df = self._apply_two_stage_refinement(result_df)

        return result_df

    def _apply_competitive_analysis(self, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply competitive field analysis to enhance predictions.

        Args:
            df_predictions: DataFrame with base predictions

        Returns:
            DataFrame with competitive-adjusted predictions
        """
        result_df = df_predictions.copy()

        # Process each race separately
        unique_races = result_df['comp'].unique()
        enhanced_dfs = []

        for race_comp in unique_races:
            race_df = result_df[result_df['comp'] == race_comp].copy()

            # Prepare base predictions for competitive analyzer
            base_predictions = {}
            if 'predicted_position_tabnet' in race_df.columns:
                base_predictions['tabnet'] = race_df['predicted_position_tabnet'].values
            if 'predicted_position_rf' in race_df.columns:
                base_predictions['rf'] = race_df['predicted_position_rf'].values
            if 'predicted_position_base' in race_df.columns:
                base_predictions['ensemble'] = race_df['predicted_position_base'].values

            # Prepare race metadata
            race_metadata = {
                'comp': race_comp,
                'hippo': race_df['hippo'].iloc[0] if 'hippo' in race_df.columns else None,
                'dist': race_df['dist'].iloc[0] if 'dist' in race_df.columns else None,
                'typec': race_df['typec'].iloc[0] if 'typec' in race_df.columns else None,
                'partant': len(race_df)
            }

            try:
                # Run competitive analysis
                competitive_results = self.competitive_analyzer.analyze_competitive_field(
                    race_df, base_predictions, race_metadata
                )

                # Extract enhanced predictions
                enhanced_predictions = competitive_results.get('enhanced_predictions', {})

                # Use the enhanced ensemble prediction if available
                if 'ensemble' in enhanced_predictions:
                    race_df['predicted_position'] = enhanced_predictions['ensemble']
                elif 'tabnet' in enhanced_predictions:
                    race_df['predicted_position'] = enhanced_predictions['tabnet']
                elif 'predicted_position_base' in race_df.columns:
                    race_df['predicted_position'] = race_df['predicted_position_base']
                else:
                    # Fallback to base prediction
                    if 'predicted_position_tabnet' in race_df.columns:
                        race_df['predicted_position'] = race_df['predicted_position_tabnet']
                    elif 'predicted_position_rf' in race_df.columns:
                        race_df['predicted_position'] = race_df['predicted_position_rf']

                # Store competitive analysis details
                if 'competitive_analysis' in competitive_results:
                    comp_analysis = competitive_results['competitive_analysis']
                    for idx, row_idx in enumerate(race_df.index):
                        if idx < len(comp_analysis):
                            horse_analysis = comp_analysis[idx]
                            race_df.at[row_idx, 'competitive_score'] = horse_analysis.get('competitive_score', 0.0)
                            race_df.at[row_idx, 'competitive_adjustment'] = horse_analysis.get('adjustment', 0.0)

            except Exception as e:
                if self.verbose:
                    self.log_info(f"  ⚠ Competitive analysis failed for race {race_comp}: {e}")
                # Use base prediction as fallback
                if 'predicted_position_base' in race_df.columns:
                    race_df['predicted_position'] = race_df['predicted_position_base']
                else:
                    race_df['predicted_position'] = race_df['predicted_position_tabnet'] if 'predicted_position_tabnet' in race_df.columns else race_df['predicted_position_rf']

            # Apply bias-based calibration (quinte model) for this race
            from core.calibration.prediction_calibrator import PredictionCalibrator
            from pathlib import Path

            calibration_path = Path('models/calibration/quinte_calibration.json')
            if calibration_path.exists():
                try:
                    calibrator = PredictionCalibrator(calibration_path)

                    # Prepare DataFrame for calibration - build it properly
                    calib_df = race_df[['predicted_position', 'numero']].copy()

                    # Add cotedirect column (with default if missing)
                    if 'cotedirect' in race_df.columns:
                        calib_df['cotedirect'] = race_df['cotedirect'].values
                    else:
                        calib_df['cotedirect'] = 5.0  # Broadcast to all rows

                    # Add race characteristics (broadcast scalars to all rows)
                    calib_df['distance'] = race_metadata.get('dist', 1600)
                    calib_df['typec'] = race_metadata.get('typec', 'P')
                    calib_df['partant'] = race_metadata.get('partant', len(race_df))

                    if self.verbose:
                        self.log_info(f"  Before calibration: {race_df['predicted_position'].head().values}")

                    # Apply bias calibration
                    calibrated_df = calibrator.transform(calib_df)
                    race_df['predicted_position_uncalibrated'] = race_df['predicted_position'].values
                    race_df['predicted_position'] = calibrated_df['calibrated_prediction'].values

                    if self.verbose:
                        self.log_info(f"  After calibration: {race_df['predicted_position'].head().values}")
                        self.log_info(f"  ✓ Applied quinté bias calibration for race {race_comp}")

                except Exception as e:
                    if self.verbose:
                        self.log_info(f"  ⚠ Quinté bias calibration failed: {e}")
                    import traceback
                    traceback.print_exc()

            enhanced_dfs.append(race_df)

        # Combine all races
        result_df = pd.concat(enhanced_dfs, ignore_index=True)

        return result_df

    def _apply_two_stage_refinement(self, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply two-stage refinement to fix positions 4-5 accuracy.

        Based on failure analysis of 66 races:
        - Positions 1-3: GOOD (74%, 69%, 51% success) - keep as-is
        - Position 4: TERRIBLE (38% success, drift +3.21) - needs fixing
        - Position 5: CHAOTIC (49% success, high variance) - needs fixing
        - Root cause: Model predicts horse quality well, but positions 4-5 depend on
          race dynamics (trips, pace) not quality. Missing longshots (odds 15-35).

        Strategy:
        - Stage 1: Keep top 3 predictions (model is good here)
        - Stage 2: Recalculate positions 4-5 with odds-based adjustments
          * Boost longshots (odds 15-35) that often finish 4-5 but predicted 6-12
          * Penalize mid-odds (10-18) at predicted 4-5 that fail 62% of time

        Args:
            df_predictions: DataFrame with predictions

        Returns:
            DataFrame with refined predictions for positions 4-5
        """
        if self.verbose:
            self.log_info("\n" + "="*80)
            self.log_info("APPLYING TWO-STAGE REFINEMENT FOR POSITIONS 4-5")
            self.log_info("="*80)

        result_df = df_predictions.copy()

        # Process each race separately
        unique_races = result_df['comp'].unique()
        refined_dfs = []

        for race_comp in unique_races:
            race_df = result_df[result_df['comp'] == race_comp].copy()

            # VALIDATION: Check required columns
            if 'cotedirect' not in race_df.columns:
                if self.verbose:
                    self.log_info(f"⚠ Race {race_comp}: Missing 'cotedirect' column, skipping two-stage refinement")
                refined_dfs.append(race_df)
                continue

            if 'predicted_position' not in race_df.columns:
                if self.verbose:
                    self.log_info(f"⚠ Race {race_comp}: Missing 'predicted_position' column, skipping two-stage refinement")
                refined_dfs.append(race_df)
                continue

            # VALIDATION: Check minimum horses
            if len(race_df) < 10:
                if self.verbose:
                    self.log_info(f"⚠ Race {race_comp}: Too few horses ({len(race_df)}), skipping two-stage refinement")
                refined_dfs.append(race_df)
                continue

            if self.verbose:
                self.log_info(f"\n--- Race {race_comp} ({len(race_df)} horses) ---")

            # Store original predicted position for logging
            race_df['original_predicted_position'] = race_df['predicted_position'].copy()

            # Sort by predicted position to identify top 3 and remaining horses
            race_df = race_df.sort_values('predicted_position').reset_index(drop=True)

            # STAGE 1: Keep top 3 as-is (model is good at predicting winners)
            top3 = race_df.iloc[:3].copy()

            # VALIDATION: Check we got exactly 3 horses
            assert len(top3) == 3, f"Stage 1 failed: got {len(top3)} horses, expected 3"

            if self.verbose:
                self.log_info("\nStage 1 - Top 3 (unchanged):")
                for idx, row in top3.iterrows():
                    horse_name = row.get('nom', 'Unknown')[:20]
                    odds = row.get('cotedirect', 0.0)
                    pos = row['predicted_position']
                    self.log_info(f"  #{row['numero']:2d}: {horse_name:20s} (odds {odds:5.1f}, pos {pos:.2f})")

            # STAGE 2: Recalculate positions 4-5 from remaining horses
            remaining = race_df.iloc[3:].copy()

            if len(remaining) < 2:
                if self.verbose:
                    self.log_info(f"⚠ Not enough horses for positions 4-5, keeping all predictions as-is")
                refined_dfs.append(race_df)
                continue

            # Create adjustment score (starts at 0)
            remaining['position_adjustment'] = 0.0

            # CRITICAL FIX 1: Boost longshots (odds 15-35)
            # Analysis shows these finish in positions 4-5 but model predicts them at 6-12
            mask_longshot = (remaining['cotedirect'] >= 15) & (remaining['cotedirect'] <= 35)
            longshots_count = mask_longshot.sum()

            if longshots_count > 0:
                # Boost by moving them UP in ranking (subtract from position)
                remaining.loc[mask_longshot, 'position_adjustment'] -= 2.5

                if self.verbose:
                    longshot_horses = remaining[mask_longshot]
                    self.log_info(f"\nLongshots boosted (odds 15-35): {longshots_count} horses")
                    for idx, row in longshot_horses.iterrows():
                        horse_name = row.get('nom', 'Unknown')[:20]
                        self.log_info(f"  #{row['numero']:2d}: {horse_name:20s} (odds {row['cotedirect']:5.1f}) - boosted -2.5")

            # CRITICAL FIX 2: Penalize mid-odds horses currently predicted at positions 4-5
            # Analysis shows odds 10-18 at predicted 4-5 fail 62% of time
            mask_mid_odds = (remaining['cotedirect'] >= 10) & (remaining['cotedirect'] <= 18)
            mask_predicted_45 = (remaining['predicted_position'] >= 4) & (remaining['predicted_position'] <= 5.5)
            mask_penalize = mask_mid_odds & mask_predicted_45
            penalized_count = mask_penalize.sum()

            if penalized_count > 0:
                # Penalize by moving them DOWN in ranking (add to position)
                remaining.loc[mask_penalize, 'position_adjustment'] += 1.5

                if self.verbose:
                    penalized_horses = remaining[mask_penalize]
                    self.log_info(f"\nMid-odds penalized (odds 10-18 at pos 4-5): {penalized_count} horses")
                    for idx, row in penalized_horses.iterrows():
                        horse_name = row.get('nom', 'Unknown')[:20]
                        self.log_info(f"  #{row['numero']:2d}: {horse_name:20s} (odds {row['cotedirect']:5.1f}, pos {row['predicted_position']:.2f}) - penalized +1.5")

            # Apply adjustments to predicted position
            remaining['predicted_position'] = remaining['predicted_position'] + remaining['position_adjustment']

            # Re-sort by adjusted predicted position and select new positions 4-5
            remaining = remaining.sort_values('predicted_position').reset_index(drop=True)
            new_45 = remaining.iloc[:2].copy()

            # VALIDATION: Check we got exactly 2 horses
            assert len(new_45) == 2, f"Stage 2 failed: got {len(new_45)} horses, expected 2"

            if self.verbose:
                self.log_info("\nStage 2 - Positions 4-5 (adjusted):")
                for idx, row in new_45.iterrows():
                    horse_name = row.get('nom', 'Unknown')[:20]
                    odds = row.get('cotedirect', 0.0)
                    old_pos = row['original_predicted_position']
                    new_pos = row['predicted_position']
                    adjustment = row['position_adjustment']
                    self.log_info(f"  #{row['numero']:2d}: {horse_name:20s} (odds {odds:5.1f}) - {old_pos:.2f} → {new_pos:.2f} (adj {adjustment:+.1f})")

            # Combine: top3 + new 4-5
            final_top5 = pd.concat([top3, new_45], ignore_index=True)

            # VALIDATION: Final checks
            assert len(final_top5) == 5, f"Final top 5 has {len(final_top5)} horses, expected 5"
            assert final_top5['numero'].duplicated().sum() == 0, f"Duplicate horses in top 5!"

            # For horses not in top 5, keep their original predictions
            not_top5_mask = ~race_df['numero'].isin(final_top5['numero'])
            not_top5 = race_df[not_top5_mask].copy()

            # Combine final top 5 with remaining horses
            # Note: We need to reassemble the full race dataframe
            # The top 5 horses have potentially new predicted_positions (for 4-5)
            # The rest keep their original positions

            # Create mapping of numero -> updated row
            updated_horses = {}
            for idx, row in final_top5.iterrows():
                updated_horses[row['numero']] = row

            # Update race_df with new predictions for top 5
            for idx, row in race_df.iterrows():
                if row['numero'] in updated_horses:
                    updated_row = updated_horses[row['numero']]
                    race_df.at[idx, 'predicted_position'] = updated_row['predicted_position']
                    if 'position_adjustment' in updated_row.index:
                        race_df.at[idx, 'position_adjustment'] = updated_row.get('position_adjustment', 0.0)

            refined_dfs.append(race_df)

        # Combine all races
        result_df = pd.concat(refined_dfs, ignore_index=True)

        if self.verbose:
            self.log_info("\n" + "="*80)
            self.log_info("TWO-STAGE REFINEMENT COMPLETE")
            self.log_info("="*80 + "\n")

        return result_df

    def format_predictions(self, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Format predictions for output.

        Args:
            df_predictions: DataFrame with predictions

        Returns:
            DataFrame with key columns formatted
        """
        # Select key columns for output
        output_columns = [
            'comp', 'jour', 'hippo', 'reun', 'prix', 'prixnom',
            'numero', 'nom', 'idche', 'age', 'sexe',
            'jockey', 'entraineur', 'cotedirect',
            'predicted_position', 'predicted_position_tabnet', 'predicted_position_rf',
            'predicted_position_uncalibrated',  # For calibration debugging
            'raw_rf_prediction', 'raw_tabnet_prediction',  # For debugging calibration compression
            'predicted_position_base',  # For debugging blending
            'original_predicted_position',  # For two-stage comparison (before refinement)
            'position_adjustment'  # For two-stage debugging (adjustment amount)
        ]

        # Only include columns that exist
        available_columns = [col for col in output_columns if col in df_predictions.columns]

        result_df = df_predictions[available_columns].copy()

        # Sort by race and predicted position
        result_df = result_df.sort_values(['comp', 'predicted_position'])

        # Add rank within each race
        result_df['predicted_rank'] = result_df.groupby('comp')['predicted_position'].rank(method='min').astype(int)

        return result_df

    def save_predictions(self, df_predictions: pd.DataFrame, output_dir: str = 'predictions') -> Dict[str, str]:
        """
        Save predictions to CSV and JSON files.

        Args:
            df_predictions: DataFrame with predictions
            output_dir: Directory to save predictions

        Returns:
            Dict with paths to saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Determine filename based on data
        unique_dates = df_predictions['jour'].nunique() if len(df_predictions) > 0 else 0

        if unique_dates == 1:
            # Single date - use date in filename
            race_date = df_predictions['jour'].iloc[0]
            filename_prefix = f"quinte_predictions_{race_date}_{timestamp}"
        else:
            # Multiple dates or all races - use "all" in filename
            min_date = df_predictions['jour'].min()
            max_date = df_predictions['jour'].max()
            filename_prefix = f"quinte_predictions_all_{min_date}_to_{max_date}_{timestamp}"

        # Save CSV
        csv_file = output_path / f"{filename_prefix}.csv"
        df_predictions.to_csv(csv_file, index=False)

        # Save JSON (grouped by race)
        json_data = []
        for race_comp, race_df in df_predictions.groupby('comp'):
            race_data = {
                'comp': race_comp,
                'jour': race_df['jour'].iloc[0],
                'hippo': race_df['hippo'].iloc[0],
                'prixnom': race_df['prixnom'].iloc[0],
                'predictions': race_df.to_dict('records')
            }
            json_data.append(race_data)

        json_file = output_path / f"{filename_prefix}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        if self.verbose:
            self.log_info(f"Saved predictions: {csv_file.name}")

        return {
            'csv': str(csv_file),
            'json': str(json_file)
        }

    def store_predictions_to_database(self, df_predictions: pd.DataFrame) -> Dict[str, int]:
        """
        Store quinté predictions to race_predictions table in database.

        Args:
            df_predictions: DataFrame with predictions for all races

        Returns:
            Dict with storage statistics (races_stored, horses_stored)
        """
        races_stored = 0
        horses_stored = 0

        # Group by race (comp)
        for race_comp, race_df in df_predictions.groupby('comp'):
            try:
                # Prepare prediction data for this race
                predictions_data = []

                for _, horse_row in race_df.iterrows():
                    horse_data = {
                        'horse_id': horse_row.get('idche'),
                        'horse_number': horse_row.get('numero'),
                        'horse_name': horse_row.get('nom'),

                        # Quinté model predictions
                        'quinte_rf_prediction': horse_row.get('predicted_position_rf'),
                        'quinte_tabnet_prediction': horse_row.get('predicted_position_tabnet'),

                        # General model predictions (if blended)
                        'general_rf_prediction': horse_row.get('general_rf_prediction'),
                        'general_tabnet_prediction': horse_row.get('general_tabnet_prediction'),

                        # Blend weights
                        'quinte_weight': self.quinte_weight,
                        'general_weight': self.general_weight,
                        'ensemble_weight_rf': 0.4,
                        'ensemble_weight_tabnet': 0.6,
                        'ensemble_prediction': horse_row.get('predicted_position_base'),

                        # Competitive analysis
                        'competitive_adjustment': horse_row.get('competitive_adjustment', 0.0),
                        'primary_advantage_type': horse_row.get('primary_advantage_type', 'none'),
                        'advantage_strength': horse_row.get('advantage_strength', 0.0),

                        # Calibration info
                        'calibrated_rf_prediction': horse_row.get('calibrated_rf_prediction'),
                        'calibrated_tabnet_prediction': horse_row.get('calibrated_tabnet_prediction'),
                        'calibration_applied': self.rf_calibrator is not None or self.tabnet_calibrator is not None,

                        # Final results
                        'final_prediction': horse_row.get('predicted_position'),
                        'predicted_rank': horse_row.get('predicted_rank'),

                        # Quinté-specific features
                        'is_favorite': horse_row.get('is_favorite', False),
                        'quinte_score': horse_row.get('quinte_score'),
                        'quinte_form_rating': horse_row.get('quinte_form_rating')
                    }
                    predictions_data.append(horse_data)

                # Create race metadata for Quinté-specific storage
                race_metadata = {
                    'race_date': race_df['jour'].iloc[0] if 'jour' in race_df.columns else None,
                    'track': race_df['hippo'].iloc[0] if 'hippo' in race_df.columns else None,
                    'race_number': race_df['prix'].iloc[0] if 'prix' in race_df.columns else None,
                    'race_name': race_df['prixnom'].iloc[0] if 'prixnom' in race_df.columns else None,
                    'predicted_quinte': ','.join(race_df.nsmallest(5, 'predicted_position')['numero'].astype(str).tolist()),
                    'predicted_winner': int(race_df.nsmallest(1, 'predicted_position')['numero'].iloc[0]),
                    'total_horses': len(race_df),
                    'quinte_weight': self.quinte_weight,
                    'general_weight': self.general_weight,
                    'calibration_applied': self.rf_calibrator is not None or self.tabnet_calibrator is not None
                }

                # Store Quinté race predictions to dedicated quinte_predictions table
                stored_count = self.prediction_storage.store_quinte_predictions(
                    race_id=race_comp,
                    predictions_data=predictions_data,
                    race_metadata=race_metadata
                )

                if stored_count > 0:
                    races_stored += 1
                    horses_stored += stored_count

            except Exception as e:
                if self.verbose:
                    self.log_info(f"  ⚠ Failed to store predictions for race {race_comp}: {e}")

        if self.verbose:
            self.log_info(f"Stored: {races_stored} races, {horses_stored} horses")

        return {
            'races_stored': races_stored,
            'horses_stored': horses_stored
        }

    def run_prediction(self, race_date: Optional[str] = None, output_dir: str = 'predictions',
                      store_to_db: bool = True) -> Dict:
        """
        Complete prediction workflow.

        Args:
            race_date: Specific date to predict (YYYY-MM-DD format), or None for ALL quinté races
            output_dir: Directory to save predictions
            store_to_db: Whether to store predictions in race_predictions table (default: True)

        Returns:
            Dict with prediction results and file paths
        """
        start_time = datetime.now()
        self.log_info("=" * 60)
        self.log_info("STARTING QUINTÉ PREDICTION")
        self.log_info("=" * 60)

        # Step 1: Load quinté races
        df_races = self.load_daily_quinte_races(race_date)

        if len(df_races) == 0:
            self.log_info("No quinté races found for the specified date")
            return {
                'status': 'no_races',
                'message': 'No quinté races found'
            }

        # Step 2: Expand participants
        df_participants = self.expand_participants(df_races)

        # Step 3: Prepare features
        df_features = self.prepare_features(df_participants)

        # Step 4: Generate predictions
        df_predictions = self.predict(df_features)

        # Step 5: Format predictions
        df_formatted = self.format_predictions(df_predictions)

        # Step 6: Save predictions to files
        saved_files = self.save_predictions(df_formatted, output_dir)

        # Step 7: Store predictions to database (NEW - replaces general predictions)
        db_storage = {}
        if store_to_db:
            db_storage = self.store_predictions_to_database(df_formatted)

        prediction_time = (datetime.now() - start_time).total_seconds()

        self.log_info("=" * 60)
        self.log_info(f"COMPLETED IN {prediction_time:.2f}s")
        self.log_info(f"  Races: {len(df_races)} | Horses: {len(df_predictions)}")
        if store_to_db and self.verbose:
            self.log_info(f"  Stored: {db_storage.get('races_stored', 0)} races, {db_storage.get('horses_stored', 0)} horses")
        self.log_info("=" * 60)

        return {
            'status': 'success',
            'prediction_time': prediction_time,
            'races': len(df_races),
            'horses': len(df_predictions),
            'files': saved_files,
            'db_storage': db_storage,
            'predictions': df_formatted
        }


def main():
    """Main entry point for quinté prediction script."""
    parser = argparse.ArgumentParser(
        description='Predict quinté races from daily_race table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict ALL quinté races in daily_race table
  python race_prediction/predict_quinte.py

  # Predict quinté races for specific date
  python race_prediction/predict_quinte.py --date 2025-10-10

  # With verbose output
  python race_prediction/predict_quinte.py --verbose
        """
    )
    parser.add_argument('--date', type=str, help='Race date (YYYY-MM-DD format). If not specified, predicts ALL quinté races in daily_race table')
    parser.add_argument('--output-dir', type=str, default='predictions', help='Output directory for predictions')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Run prediction
    engine = QuintePredictionEngine(verbose=args.verbose)
    result = engine.run_prediction(race_date=args.date, output_dir=args.output_dir)

    if result['status'] == 'success':
        print("\n✓ Prediction complete!")
        print(f"  CSV: {result['files']['csv']}")
        print(f"  JSON: {result['files']['json']}")
    else:
        print(f"\n✗ Prediction failed: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
