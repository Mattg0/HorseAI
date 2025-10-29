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
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from utils.env_setup import AppConfig, get_sqlite_dbpath
from utils.model_manager import ModelManager
from core.calculators.static_feature_calculator import FeatureCalculator
from core.calculators.quinte_feature_calculator import QuinteFeatureCalculator
from race_prediction.competitive_field_analyzer import CompetitiveFieldAnalyzer
from race_prediction.simple_prediction_storage import SimplePredictionStorage


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

        self._load_models()

        # Initialize quinté feature calculator
        self.quinte_calculator = QuinteFeatureCalculator(self.db_path)

        # Initialize competitive field analyzer
        self.competitive_analyzer = CompetitiveFieldAnalyzer(verbose=self.verbose, db_path=self.db_path)

        # Initialize prediction storage
        self.prediction_storage = SimplePredictionStorage(db_path=self.db_path, verbose=self.verbose)

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
            # Load Random Forest quinté model
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
                self.log_info(f"✓ Loaded TabNet quinté model from {tabnet_info['path']}")
                self.log_info(f"  Features: {len(self.feature_columns)}")

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

        # DEBUG: Check if pre-calculated features exist
        if len(df_participants) > 0:
            sample = df_participants.iloc[0]
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG - After expand_participants:")
            print(f"{'='*80}")
            print(f"  che_bytype_dnf_rate: {sample.get('che_bytype_dnf_rate', 'MISSING')}")
            print(f"  victoirescheval: {sample.get('victoirescheval', 'MISSING')}")
            print(f"  placescheval: {sample.get('placescheval', 'MISSING')}")
            print(f"  musiqueche: {str(sample.get('musiqueche', 'MISSING'))[:50]}")
            print(f"  typec: {sample.get('typec', 'MISSING')}")
            print(f"{'='*80}\n")

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

        # DEBUG: Check if FeatureCalculator overwrote the pre-calculated features
        if len(df_with_features) > 0:
            sample = df_with_features.iloc[0]
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG - After FeatureCalculator:")
            print(f"{'='*80}")
            print(f"  che_bytype_dnf_rate: {sample.get('che_bytype_dnf_rate', 'MISSING')}")
            print(f"  victoirescheval: {sample.get('victoirescheval', 'MISSING')}")
            print(f"  placescheval: {sample.get('placescheval', 'MISSING')}")
            print(f"  musiqueche: {str(sample.get('musiqueche', 'MISSING'))[:50]}")
            print(f"  typec: {sample.get('typec', 'MISSING')}")
            print(f"{'='*80}\n")

        # Step 2: Batch-load all quinté historical data
        self.log_info("Batch-loading quinté historical data...")
        all_quinte_data = self.quinte_calculator.batch_load_all_quinte_data()

        self.log_info(f"Loaded {len(all_quinte_data['races'])} historical races")

        # Step 3: Add quinté features using batch-loaded data
        self.log_info("Calculating quinté features...")

        unique_races = df_with_features['comp'].unique()

        # Process each race separately to add quinté features
        race_dfs = []
        for idx, race_comp in enumerate(unique_races):
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

        # Save features to JSON for debugging/analysis
        features_path = './features.json'
        df_complete.to_json(features_path, orient='records', indent=2)
        self.log_info(f"Saved features to: {features_path}")

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
        # feature_columns has recence_log/cotedirect_log, but result_df has recence/cotedirect
        transform_mapping = {
            'recence_log': 'recence',
            'cotedirect_log': 'cotedirect'
        }

        # Select features - map transformed names back to original names for selection
        available_features = []
        missing_features = []
        for col in self.feature_columns:
            # Check if this is a transformed feature
            original_name = transform_mapping.get(col, col)
            if original_name in result_df.columns:
                available_features.append(original_name)
            else:
                missing_features.append(col)

        if missing_features:
            self.log_info(f"⚠️  Warning: {len(missing_features)} features missing from dataframe:")
            self.log_info(f"  {missing_features[:10]}")

        X = result_df[available_features].copy()

        # SKIP clean_features() - training doesn't use it, so prediction shouldn't either
        # We want ALL 90 features including bytype, global, etc.
        # X = cleaner.clean_features(X)  # ← DISABLED

        # Apply transformations (recence→recence_log, cotedirect→cotedirect_log)
        # This matches what was done during training
        X = cleaner.apply_transformations(X)
        X = X.fillna(0)

        self.log_info(f"✓ Final feature matrix: {len(X.columns)} features (model expects {len(self.feature_columns)})")

        # FAIL-FAST: Validate exact feature alignment before calling scaler
        if len(X.columns) != len(self.feature_columns):
            missing_features = set(self.feature_columns) - set(X.columns)
            extra_features = set(X.columns) - set(self.feature_columns)

            # Save detailed comparison to file for debugging
            with open('feature_mismatch_debug.txt', 'w') as f:
                f.write(f"FEATURE MISMATCH DEBUG\n")
                f.write(f"=" * 80 + "\n\n")
                f.write(f"Created: {len(X.columns)} features\n")
                f.write(f"Expected: {len(self.feature_columns)} features\n\n")
                f.write(f"Missing from X ({len(missing_features)} features):\n")
                for feat in sorted(missing_features):
                    f.write(f"  - {feat}\n")
                f.write(f"\nExtra in X ({len(extra_features)} features):\n")
                for feat in sorted(extra_features):
                    f.write(f"  - {feat}\n")
                f.write(f"\nAll created features ({len(X.columns)}):\n")
                for feat in sorted(X.columns):
                    f.write(f"  - {feat}\n")
                f.write(f"\nAll expected features ({len(self.feature_columns)}):\n")
                for feat in sorted(self.feature_columns):
                    f.write(f"  - {feat}\n")

            mismatch_msg = f"""
❌ FEATURE COUNT MISMATCH!

Created: {len(X.columns)} features
Expected: {len(self.feature_columns)} features

Missing from X: {sorted(missing_features)}
Extra in X: {sorted(extra_features)}

Detailed comparison saved to: feature_mismatch_debug.txt

This indicates the feature calculation or transformation is not matching the training pipeline.
"""
            self.log_info(mismatch_msg)
            raise ValueError(f"Feature count mismatch: {len(X.columns)} != {len(self.feature_columns)}")

        # Validate exact feature names match
        X_features = set(X.columns)
        model_features = set(self.feature_columns)
        if X_features != model_features:
            missing = model_features - X_features
            extra = X_features - model_features
            mismatch_msg = f"""
❌ FEATURE NAME MISMATCH!

Missing: {missing}
Extra: {extra}

Feature calculation is creating different features than training!
"""
            self.log_info(mismatch_msg)
            raise ValueError("Feature names don't match model expectations")

        features_path = './X_predict_feature.json'
        X.to_json(features_path, orient='records', indent=2)
        self.log_info(f"Saved features to: {features_path}")

        # TabNet predictions
        if self.tabnet_model and self.scaler:
            self.log_info("  Running TabNet model...")
            # Ensure column order matches scaler's expected order
            if hasattr(self.scaler, 'feature_names_in_'):
                X = X[self.scaler.feature_names_in_]
            X_scaled = self.scaler.transform(X)
            X_scaled = X_scaled.astype(np.float32)
            tabnet_preds = self.tabnet_model.predict(X_scaled).flatten()
            result_df['predicted_position_tabnet'] = tabnet_preds

        # Random Forest predictions
        if self.rf_model:
            self.log_info("  Running Random Forest model...")
            rf_preds = self.rf_model.predict(X)
            result_df['predicted_position_rf'] = rf_preds

        # Ensemble prediction (average of both models if available)
        if 'predicted_position_tabnet' in result_df.columns and 'predicted_position_rf' in result_df.columns:
            result_df['predicted_position_base'] = (
                result_df['predicted_position_tabnet'] * 1.0 +  # TabNet weight
                result_df['predicted_position_rf'] * 0.0         # RF weight
            )
        elif 'predicted_position_tabnet' in result_df.columns:
            result_df['predicted_position_base'] = result_df['predicted_position_tabnet']
        elif 'predicted_position_rf' in result_df.columns:
            result_df['predicted_position_base'] = result_df['predicted_position_rf']
        else:
            raise ValueError("No predictions generated")

        self.log_info(f"Base predictions generated for {len(result_df)} participants")

        # Apply competitive field analysis to enhance predictions
        self.log_info("Applying competitive field analysis...")
        result_df = self._apply_competitive_analysis(result_df)

        self.log_info(f"Final predictions with competitive analysis complete for {len(result_df)} participants")

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

                if self.verbose:
                    self.log_info(f"  ✓ Competitive analysis applied to race {race_comp}")

            except Exception as e:
                self.log_info(f"  ⚠ Competitive analysis failed for race {race_comp}: {e}")
                # Use base prediction as fallback
                if 'predicted_position_base' in race_df.columns:
                    race_df['predicted_position'] = race_df['predicted_position_base']
                else:
                    race_df['predicted_position'] = race_df['predicted_position_tabnet'] if 'predicted_position_tabnet' in race_df.columns else race_df['predicted_position_rf']

            enhanced_dfs.append(race_df)

        # Combine all races
        result_df = pd.concat(enhanced_dfs, ignore_index=True)

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
            'predicted_position', 'predicted_position_tabnet', 'predicted_position_rf'
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
        self.log_info(f"✓ Saved predictions to {csv_file}")

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
        self.log_info(f"✓ Saved predictions to {json_file}")

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
        self.log_info("\nStoring predictions to database...")

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
                        'rf_prediction': horse_row.get('predicted_position_rf'),
                        'tabnet_prediction': horse_row.get('predicted_position_tabnet'),
                        'ensemble_weight_rf': 0.4,  # Default quinté weights
                        'ensemble_weight_tabnet': 0.6,
                        'ensemble_prediction': horse_row.get('predicted_position_base'),
                        'competitive_adjustment': horse_row.get('competitive_adjustment', 0.0),
                        'primary_advantage_type': horse_row.get('primary_advantage_type', 'none'),
                        'advantage_strength': horse_row.get('advantage_strength', 0.0),
                        'final_prediction': horse_row.get('predicted_position')
                    }
                    predictions_data.append(horse_data)

                # Store race predictions
                stored_count = self.prediction_storage.store_race_predictions(
                    race_id=race_comp,
                    predictions_data=predictions_data
                )

                if stored_count > 0:
                    races_stored += 1
                    horses_stored += stored_count
                    self.log_info(f"  ✓ Stored {stored_count} predictions for race {race_comp}")

            except Exception as e:
                self.log_info(f"  ⚠ Failed to store predictions for race {race_comp}: {e}")

        self.log_info(f"\n✓ Database storage complete: {races_stored} races, {horses_stored} horses")

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
        self.log_info(f"QUINTÉ PREDICTION COMPLETED IN {prediction_time:.2f}s")
        self.log_info(f"Races predicted: {len(df_races)}")
        self.log_info(f"Horses predicted: {len(df_predictions)}")
        if store_to_db:
            self.log_info(f"Database storage: {db_storage.get('races_stored', 0)} races, {db_storage.get('horses_stored', 0)} horses")
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
