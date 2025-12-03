#!/usr/bin/env python3
"""
Quinté Incremental Training Pipeline

Implements incremental training specifically for quinté prediction models.
Extends the IncrementalTrainingPipeline approach for quinté-specific use cases.

Key Features:
- Fetches completed quinté races with predictions and results
- Analyzes failures using QuinteErrorAnalyzer
- Applies weighted training based on failure severity
- Fine-tunes both RF and TabNet models
- Validates improvement on quinté-specific metrics
- Saves improved models with versioning

Success Criteria:
- Quinté désordre success rate improves >= 5%
- Bonus 3/4 rates maintain or improve
- MAE stays within 5% of baseline
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TabNet imports
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

from utils.env_setup import AppConfig, get_sqlite_dbpath
from utils.model_manager import ModelManager
from core.calculators.static_feature_calculator import FeatureCalculator
from core.calculators.quinte_feature_calculator import QuinteFeatureCalculator, add_quinte_features_to_dataframe
from .quinte_error_analyzer import QuinteErrorAnalyzer


class QuinteIncrementalTrainer:
    """
    Incremental training pipeline specifically for quinté predictions.
    """

    def __init__(self, model_path: str = None, db_name: str = None,
                 output_dir: str = None, verbose: bool = False):
        """
        Initialize the quinté incremental trainer.

        Args:
            model_path: Path to base quinté models (if None, uses latest)
            db_name: Database name from config (defaults to active_db)
            output_dir: Directory for analysis outputs
            verbose: Whether to print verbose output
        """
        # Initialize config
        self.config = AppConfig()
        self.verbose = verbose

        # Set database
        if db_name is None:
            self.db_name = self.config._config.base.active_db
        else:
            self.db_name = db_name

        # Get database path
        self.db_path = get_sqlite_dbpath(self.db_name)

        # Initialize model manager
        self.model_manager = ModelManager()

        # Get model paths - look for quinté-specific models
        if model_path is None:
            all_paths = self.model_manager.get_all_model_paths()
            # Try to find quinté models first
            self.rf_model_path = all_paths.get('quinte_rf') or all_paths.get('rf')
            self.tabnet_model_path = all_paths.get('quinte_tabnet') or all_paths.get('tabnet')
        else:
            model_path = Path(model_path)
            self.rf_model_path = model_path
            self.tabnet_model_path = model_path

        # Setup output directory
        if output_dir is None:
            self.output_dir = Path("incremental_training/quinte") / datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.error_analyzer = QuinteErrorAnalyzer(verbose=verbose)
        self.quinte_calculator = QuinteFeatureCalculator(self.db_path)

        # Load existing models
        self._load_models()

        # Improvement thresholds
        self.improvement_thresholds = {
            'quinte_desordre': 0.05,  # 5% improvement
            'bonus_4': 0.05,
            'bonus_3': 0.05,
            'mae': 0.05  # Can increase up to 5%
        }

        if self.verbose:
            print(f"QuinteIncrementalTrainer initialized")
            print(f"  RF Model: {self.rf_model_path}")
            print(f"  TabNet Model: {self.tabnet_model_path}")
            print(f"  Database: {self.db_path}")
            print(f"  Output: {self.output_dir}")

    def _load_models(self):
        """Load base quinté models."""
        self.rf_model = None
        self.tabnet_model = None
        self.tabnet_scaler = None
        self.feature_columns = None

        # Load RF model
        if self.rf_model_path and Path(self.rf_model_path).exists():
            rf_file = Path(self.rf_model_path) / 'rf_model.joblib'
            if rf_file.exists():
                try:
                    self.rf_model = joblib.load(rf_file)
                    if self.verbose:
                        print(f"Loaded RF model from: {rf_file}")
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading RF model: {e}")

        # Load TabNet model
        if TABNET_AVAILABLE and self.tabnet_model_path and Path(self.tabnet_model_path).exists():
            tabnet_file = Path(self.tabnet_model_path) / 'tabnet_model.zip'
            if tabnet_file.exists():
                try:
                    self.tabnet_model = TabNetRegressor()
                    self.tabnet_model.load_model(str(tabnet_file))

                    # Load scaler
                    scaler_file = Path(self.tabnet_model_path) / 'tabnet_scaler.joblib'
                    if scaler_file.exists():
                        self.tabnet_scaler = joblib.load(scaler_file)

                    # Load feature columns
                    config_file = Path(self.tabnet_model_path) / 'tabnet_config.json'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            self.feature_columns = config.get('feature_columns', [])

                    if self.verbose:
                        print(f"Loaded TabNet model from: {tabnet_file}")
                        if self.feature_columns:
                            print(f"TabNet expects {len(self.feature_columns)} features")

                except Exception as e:
                    if self.verbose:
                        print(f"Error loading TabNet model: {e}")

    def get_completed_quinte_races(self, date_from: str, date_to: str,
                                   limit: int = None, use_db_predictions: bool = True) -> List[Dict]:
        """
        Fetch completed quinté races with predictions and actual results.

        Args:
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            limit: Maximum number of races to fetch
            use_db_predictions: If True, fetch from race_predictions table (NEW).
                               If False, use legacy prediction_results JSON column.

        Returns:
            List of race dictionaries with predictions embedded
        """
        if self.verbose:
            source = "race_predictions table" if use_db_predictions else "prediction_results JSON"
            print(f"Fetching completed quinté races from {date_from} to {date_to} (using {source})")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if use_db_predictions:
            # NEW: Fetch races that have predictions in race_predictions table
            query = """
            SELECT DISTINCT
                dr.*,
                COUNT(rp.id) as prediction_count
            FROM daily_race dr
            INNER JOIN race_predictions rp ON dr.comp = rp.race_id
            WHERE dr.quinte = 1
            AND dr.actual_results IS NOT NULL
            AND dr.actual_results != 'pending'
            AND dr.jour BETWEEN ? AND ?
            GROUP BY dr.comp
            HAVING prediction_count > 0
            ORDER BY dr.jour DESC
            """
        else:
            # LEGACY: Use prediction_results JSON column
            query = """
            SELECT * FROM daily_race
            WHERE quinte = 1
            AND actual_results IS NOT NULL
            AND actual_results != 'pending'
            AND prediction_results IS NOT NULL
            AND jour BETWEEN ? AND ?
            ORDER BY jour DESC
            """

        params = [date_from, date_to]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to list of dictionaries
        races = [dict(row) for row in rows]

        if use_db_predictions:
            # Fetch predictions from race_predictions table and embed them
            for race in races:
                race_id = race['comp']

                # Parse partants JSON to get numero mapping (idche -> numero)
                partants_json = race.get('partants', '[]')
                if isinstance(partants_json, str):
                    try:
                        partants = json.loads(partants_json)
                    except:
                        partants = []
                else:
                    partants = partants_json

                # Create idche -> numero mapping
                idche_to_numero = {}
                for partant in partants:
                    idche = partant.get('idche')
                    numero = partant.get('numero')
                    if idche and numero:
                        idche_to_numero[idche] = numero

                # Get predictions for this race
                cursor.execute("""
                    SELECT
                        horse_id as idche,
                        rf_prediction as predicted_position_rf,
                        tabnet_prediction as predicted_position_tabnet,
                        ensemble_prediction as predicted_position_base,
                        final_prediction as predicted_position,
                        competitive_adjustment,
                        primary_advantage_type,
                        advantage_strength
                    FROM race_predictions
                    WHERE race_id = ?
                    ORDER BY final_prediction ASC
                """, (race_id,))

                predictions = []
                for row in cursor.fetchall():
                    pred_dict = dict(row)
                    # Add numero from partants mapping
                    idche = pred_dict.get('idche')
                    pred_dict['numero'] = idche_to_numero.get(idche, 0)
                    predictions.append(pred_dict)

                # Embed predictions as JSON (same format as prediction_results)
                race['prediction_results'] = json.dumps(predictions)

                if self.verbose and len(races) <= 5:  # Show details for first few races
                    print(f"  Race {race_id}: {len(predictions)} predictions from database (with numero mapping)")

        conn.close()

        if self.verbose:
            print(f"Found {len(races)} completed quinté races")

        return races

    def extract_failure_data(self, races: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Extract and analyze failure data from quinté races.

        Args:
            races: List of race dictionaries from database

        Returns:
            Tuple of (training_dataframe, race_analyses)
                - training_dataframe: DataFrame with features and weights
                - race_analyses: List of analysis dictionaries from error analyzer
        """
        if self.verbose:
            print(f"Analyzing {len(races)} quinté races for failures...")

        all_features = []
        race_analyses = []

        for race in races:
            try:
                # Parse prediction results
                pred_results = race.get('prediction_results')
                if isinstance(pred_results, str):
                    pred_results = json.loads(pred_results)

                # Get predicted top 5 (by predicted position)
                if isinstance(pred_results, list):
                    # Sort by predicted position and get top 5
                    sorted_preds = sorted(pred_results, key=lambda x: x.get('predicted_position', 999))
                    predicted_top5 = [p.get('numero', 0) for p in sorted_preds[:5]]
                else:
                    if self.verbose:
                        print(f"Skipping race {race.get('comp')}: Invalid prediction format")
                    continue

                # Create predictions DataFrame for analysis
                predictions_df = pd.DataFrame(pred_results)

                # Prepare race data for error analysis
                race_data = {
                    'race_id': race.get('comp'),
                    'predicted_top5': predicted_top5,
                    'actual_results': race.get('actual_results'),
                    'predictions_df': predictions_df,
                    'race_metadata': {
                        'field_size': race.get('partant', 0),
                        'track_condition': race.get('pistegp', 'unknown'),
                        'hippo': race.get('hippo', ''),
                        'date': race.get('jour', '')
                    }
                }

                # Analyze this race
                analysis = self.error_analyzer.analyze_race_prediction(race_data)
                analysis['race_id'] = race.get('comp')
                race_analyses.append(analysis)

                # Extract features for training
                # Add failure weight to each horse's features
                failure_weight = analysis['failure_weight']

                for _, horse in predictions_df.iterrows():
                    horse_features = horse.to_dict()
                    horse_features['race_id'] = race.get('comp')
                    horse_features['failure_weight'] = failure_weight
                    horse_features['jour'] = race.get('jour')
                    all_features.append(horse_features)

            except Exception as e:
                if self.verbose:
                    print(f"Error processing race {race.get('comp')}: {e}")
                continue

        # Create DataFrame
        training_df = pd.DataFrame(all_features)

        if self.verbose:
            print(f"Extracted features for {len(training_df)} horses from {len(races)} races")

            # Show failure type distribution
            failure_types = {}
            for analysis in race_analyses:
                ft = analysis['failure_type']
                failure_types[ft] = failure_types.get(ft, 0) + 1

            print("\nFailure Type Distribution:")
            for ft, count in failure_types.items():
                print(f"  {ft}: {count} ({count/len(race_analyses)*100:.1f}%)")

        return training_df, race_analyses

    def calculate_baseline_metrics(self, races: List[Dict]) -> Dict:
        """
        Calculate baseline quinté metrics before incremental training.

        Args:
            races: List of race dictionaries

        Returns:
            Dictionary with baseline metrics
        """
        if not races:
            return {}

        _, race_analyses = self.extract_failure_data(races)

        total_races = len(race_analyses)
        quinte_desordre_wins = sum(1 for r in race_analyses if r['quinte_desordre'])
        bonus_4_wins = sum(1 for r in race_analyses if r['bonus_4'])
        bonus_3_wins = sum(1 for r in race_analyses if r['bonus_3'])
        avg_mae = np.mean([r['mae'] for r in race_analyses])

        return {
            'total_races': total_races,
            'quinte_desordre_rate': quinte_desordre_wins / total_races if total_races > 0 else 0,
            'bonus_4_rate': bonus_4_wins / total_races if total_races > 0 else 0,
            'bonus_3_rate': bonus_3_wins / total_races if total_races > 0 else 0,
            'avg_mae': float(avg_mae)
        }

    def train_on_failures(self, training_df: pd.DataFrame, focus_on_failures: bool = True) -> Dict:
        """
        Train incremental models on failure data.

        Args:
            training_df: DataFrame with features and failure_weight
            focus_on_failures: If True, use only failures; otherwise balanced dataset

        Returns:
            Training results dictionary
        """
        if training_df.empty:
            return {'status': 'error', 'message': 'No training data available'}

        if self.verbose:
            print(f"\nTraining on {len(training_df)} samples...")

        # Prepare features (same as original quinté training)
        # Assuming training_df already has the necessary features

        # Get feature columns (exclude metadata, target, and non-numeric columns)
        exclude_cols = ['race_id', 'failure_weight', 'jour', 'actual_position', 'predicted_position', 'numero', 'cheval',
                       'primary_advantage_type', 'idche']  # Exclude string/categorical columns
        feature_cols = [col for col in training_df.columns if col not in exclude_cols]

        X = training_df[feature_cols].copy()

        # CRITICAL: Filter out any remaining non-numeric columns
        # This prevents "could not convert string to float: 'none'" errors
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            if self.verbose:
                print(f"  Filtering out {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
            X = X.select_dtypes(include=['number'])

        # Fill missing values with 0
        X = X.fillna(0)

        y = training_df['actual_position'] if 'actual_position' in training_df.columns else training_df['predicted_position']

        # Get sample weights
        sample_weights = training_df['failure_weight'].values

        # Split data
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42
        )

        results = {
            'status': 'success',
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'rf_results': {},
            'tabnet_results': {}
        }

        # Train RF model (if available)
        if self.rf_model is not None:
            if self.verbose:
                print("Training incremental RF model...")

            # Extract base estimator if model is wrapped in CalibratedRegressor
            from sklearn.calibration import CalibratedClassifierCV
            base_model = self.rf_model
            if hasattr(self.rf_model, 'estimator') or hasattr(self.rf_model, 'base_estimator'):
                # It's a calibrated model, extract base estimator
                base_model = getattr(self.rf_model, 'estimator', None) or getattr(self.rf_model, 'base_estimator', None)
                if self.verbose:
                    print(f"  Detected wrapped model, using base estimator: {type(base_model).__name__}")

            # Get parameters from base model
            try:
                n_estimators = getattr(base_model, 'n_estimators', 100)
                max_depth = getattr(base_model, 'max_depth', None)
            except:
                n_estimators = 100
                max_depth = None

            # Use existing model and continue training (warm_start for RF)
            # For RandomForest, we can't truly do incremental, so we retrain
            rf_incremental = RandomForestRegressor(
                n_estimators=n_estimators + 50,  # Add more trees
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

            rf_incremental.fit(X_train, y_train, sample_weight=w_train)

            # Validate
            rf_pred_val = rf_incremental.predict(X_val)
            rf_mae_val = mean_absolute_error(y_val, rf_pred_val)

            results['rf_results'] = {
                'validation_mae': float(rf_mae_val),
                'model': rf_incremental
            }

            if self.verbose:
                print(f"  RF Validation MAE: {rf_mae_val:.3f}")

        # Train TabNet model (if available)
        if TABNET_AVAILABLE and self.tabnet_model is not None:
            if self.verbose:
                print("Fine-tuning TabNet model...")

            # Prepare data for TabNet
            X_train_np = X_train.values.astype(np.float32)
            X_val_np = X_val.values.astype(np.float32)
            y_train_np = y_train.values.reshape(-1, 1).astype(np.float32)
            y_val_np = y_val.values.reshape(-1, 1).astype(np.float32)

            # Fine-tune existing TabNet model
            try:
                self.tabnet_model.fit(
                    X_train_np, y_train_np,
                    eval_set=[(X_val_np, y_val_np)],
                    max_epochs=20,  # Limited epochs for fine-tuning
                    patience=5,
                    batch_size=256,
                    virtual_batch_size=128,
                    weights=w_train.astype(np.float32),  # Use failure weights
                    from_unsupervised=None
                )

                # Validate
                tabnet_pred_val = self.tabnet_model.predict(X_val_np)
                tabnet_mae_val = mean_absolute_error(y_val, tabnet_pred_val)

                results['tabnet_results'] = {
                    'validation_mae': float(tabnet_mae_val),
                    'model': self.tabnet_model
                }

                if self.verbose:
                    print(f"  TabNet Validation MAE: {tabnet_mae_val:.3f}")

            except Exception as e:
                if self.verbose:
                    print(f"Error training TabNet: {e}")
                results['tabnet_results'] = {'error': str(e)}

        return results

    def save_incremental_quinte_model(self, training_results: Dict,
                                     baseline_metrics: Dict,
                                     improved_metrics: Dict) -> str:
        """
        Save improved quinté models with versioning.

        Args:
            training_results: Results from train_on_failures()
            baseline_metrics: Baseline quinté metrics
            improved_metrics: Improved quinté metrics after training

        Returns:
            Path to saved models
        """
        # Create version directory
        timestamp = datetime.now().strftime('%Y-%m-%d/%Hquinte_incremental_%H%M%S')
        save_dir = Path('models') / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"\nSaving incremental quinté models to: {save_dir}")

        # Save RF model
        if 'rf_results' in training_results and 'model' in training_results['rf_results']:
            rf_save_path = save_dir / 'rf_model.joblib'
            joblib.dump(training_results['rf_results']['model'], rf_save_path)
            if self.verbose:
                print(f"  Saved RF model: {rf_save_path}")

        # Save TabNet model
        if 'tabnet_results' in training_results and 'model' in training_results['tabnet_results']:
            tabnet_save_path = save_dir / 'tabnet_model.zip'
            training_results['tabnet_results']['model'].save_model(str(tabnet_save_path))

            # Save scaler if available
            if self.tabnet_scaler:
                scaler_path = save_dir / 'tabnet_scaler.joblib'
                joblib.dump(self.tabnet_scaler, scaler_path)

            # Save config
            if self.feature_columns:
                config_path = save_dir / 'tabnet_config.json'
                with open(config_path, 'w') as f:
                    json.dump({
                        'feature_columns': self.feature_columns,
                        'incremental_training': True,
                        'baseline_metrics': baseline_metrics,
                        'improved_metrics': improved_metrics,
                        'improvement': {
                            'quinte_desordre': improved_metrics.get('quinte_desordre_rate', 0) - baseline_metrics.get('quinte_desordre_rate', 0),
                            'bonus_4': improved_metrics.get('bonus_4_rate', 0) - baseline_metrics.get('bonus_4_rate', 0),
                            'bonus_3': improved_metrics.get('bonus_3_rate', 0) - baseline_metrics.get('bonus_3_rate', 0),
                        }
                    }, f, indent=2)

            if self.verbose:
                print(f"  Saved TabNet model: {tabnet_save_path}")

        # Save metadata
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'created_at': datetime.now().isoformat(),
                'model_type': 'quinte_incremental',
                'baseline_metrics': baseline_metrics,
                'improved_metrics': improved_metrics,
                'training_results': {
                    k: v for k, v in training_results.items() if k != 'rf_results' and k != 'tabnet_results'
                }
            }, f, indent=2)

        if self.verbose:
            print(f"  Saved metadata: {metadata_path}")

        # Update config.yaml to point to the new models
        self._update_config_with_new_models(save_dir)

        return str(save_dir)

    def _update_config_with_new_models(self, save_dir: Path):
        """Update config.yaml with paths to newly saved quinté models."""
        try:
            import yaml

            config_path = Path('config.yaml')
            if not config_path.exists():
                if self.verbose:
                    print("  ⚠ config.yaml not found, skipping config update")
                return

            # Read current config
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Ensure models section exists
            if 'models' not in config_data:
                config_data['models'] = {}
            if 'latest_models' not in config_data['models']:
                config_data['models']['latest_models'] = {}

            # Get relative path from models directory
            models_dir = Path('models')
            try:
                relative_path = save_dir.relative_to(models_dir)
            except ValueError:
                # save_dir is not relative to models_dir, use as-is
                relative_path = save_dir

            relative_path_str = str(relative_path).replace('\\', '/')

            # Update quinté model paths if the model files exist
            if (save_dir / 'rf_model.joblib').exists():
                config_data['models']['latest_models']['rf_quinte'] = relative_path_str
                if self.verbose:
                    print(f"  ✓ Updated config.yaml: rf_quinte = {relative_path_str}")

            if (save_dir / 'tabnet_model.zip').exists():
                config_data['models']['latest_models']['tabnet_quinte'] = relative_path_str
                if self.verbose:
                    print(f"  ✓ Updated config.yaml: tabnet_quinte = {relative_path_str}")

            # Write updated config
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            if self.verbose:
                print(f"  ✓ Config updated successfully - new quinté models are now active!")

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Warning: Could not update config.yaml: {e}")
            # Don't fail the save operation if config update fails


if __name__ == "__main__":
    # Example usage
    trainer = QuinteIncrementalTrainer(verbose=True)

    # Fetch recent quinté races
    from datetime import timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    races = trainer.get_completed_quinte_races(start_date, end_date, limit=20)
    print(f"\nFound {len(races)} quinté races")

    if races:
        # Calculate baseline
        baseline = trainer.calculate_baseline_metrics(races)
        print(f"\nBaseline Metrics:")
        print(f"  Quinté Désordre Rate: {baseline['quinte_desordre_rate']*100:.1f}%")
        print(f"  Bonus 4 Rate: {baseline['bonus_4_rate']*100:.1f}%")
        print(f"  Bonus 3 Rate: {baseline['bonus_3_rate']*100:.1f}%")
        print(f"  Average MAE: {baseline['avg_mae']:.3f}")
