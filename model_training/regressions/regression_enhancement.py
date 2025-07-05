# model_training/regressions/regression_enhancement.py

import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from utils.model_manager import get_model_manager
from core.orchestrators.race_archiver import RaceArchiver


class IncrementalTrainingPipeline:
    """
    Incremental training pipeline that:
    1. Fetches completed races from daily_race table (with predictions and results)
    2. Performs regression analysis and model improvement
    3. Creates new incremental model if improvements are found
    4. Archives races from daily_race to historical_races upon success
    """

    def __init__(self, model_path: str = None, db_name: str = None,
                 output_dir: str = None, verbose: bool = False):
        """
        Initialize the incremental training pipeline.

        Args:
            model_path: Path to the base model (if None, uses latest from config)
            db_name: Database name from config (defaults to active_db)
            output_dir: Directory for analysis outputs
            verbose: Whether to print verbose output
        """
        # Initialize config and database
        self.config = AppConfig()
        self.verbose = verbose

        # Set database
        if db_name is None:
            self.db_name = self.config._config.base.active_db
        else:
            self.db_name = db_name

        # Get database path
        self.db_path = self.config.get_sqlite_dbpath(self.db_name)

        # Initialize orchestrator for data processing
        self.orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=self.db_path,
            verbose=verbose
        )

        # Initialize race archiver for moving races
        self.race_archiver = RaceArchiver(db_name=db_name, verbose=verbose)

        # Get model manager and path
        self.model_manager = get_model_manager()
        if model_path is None:
            self.model_path = self.model_manager.get_model_path()
        else:
            self.model_path = Path(model_path)

        # Setup output directory
        if output_dir is None:
            self.output_dir = Path("incremental_training") / datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing models
        self._load_models()

        # Initialize containers
        self.improvement_threshold = 0.05  # 5% improvement required to create new model
        self.processed_races = []

        if self.verbose:
            print(f"IncrementalTrainingPipeline initialized")
            print(f"  Base Model: {self.model_path}")
            print(f"  Database: {self.db_path}")
            print(f"  Output: {self.output_dir}")

    def _load_models(self):
        """Load base models and configuration."""
        if self.model_path is None:
            if self.verbose:
                print("No base model path available")
            self.rf_model = None
            self.lstm_model = None
            self.model_config = {}
            return

        if self.verbose:
            print(f"Loading base models from: {self.model_path}")

        try:
            models = self.model_manager.load_models(str(self.model_path))

            self.rf_model = models.get('rf_model')
            self.lstm_model = models.get('lstm_model')
            self.model_config = models.get('model_config', {})

            if self.verbose:
                print(f"Loaded models: RF={self.rf_model is not None}, LSTM={self.lstm_model is not None}")

        except Exception as e:
            if self.verbose:
                print(f"Error loading models: {e}")
            self.rf_model = None
            self.lstm_model = None
            self.model_config = {}

    def fetch_completed_races(self, date_from: str = None, date_to: str = None,
                              limit: int = None) -> List[Dict]:
        """
        Fetch completed races from daily_race table that have both predictions and results.

        Args:
            date_from: Start date for fetching races (YYYY-MM-DD)
            date_to: End date for fetching races (YYYY-MM-DD)
            limit: Maximum number of races to fetch

        Returns:
            List of race dictionaries with predictions and results
        """
        if self.verbose:
            print(f"Fetching completed races from {date_from} to {date_to}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query for races with both predictions and actual results
        query = """
        SELECT * FROM daily_race 
        WHERE actual_results IS NOT NULL 
        AND actual_results != 'pending'
        AND prediction_results IS NOT NULL
        """

        params = []
        if date_from:
            query += " AND jour >= ?"
            params.append(date_from)
        if date_to:
            query += " AND jour <= ?"
            params.append(date_to)

        query += " ORDER BY jour DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # Convert to list of dictionaries
        races = [dict(row) for row in rows]

        if self.verbose:
            print(f"Found {len(races)} completed races with predictions and results")

        return races

    def extract_training_data(self, races: List[Dict]) -> pd.DataFrame:
        """
        Extract training data from daily races for incremental learning.

        Args:
            races: List of race dictionaries from daily_race table

        Returns:
            DataFrame with features, predictions, and actual results
        """
        if self.verbose:
            print("Extracting training data from daily races...")

        training_samples = []

        for race in races:
            try:
                # Parse JSON fields
                participants = json.loads(race['participants']) if isinstance(race['participants'], str) else race[
                    'participants']
                prediction_results = json.loads(race['prediction_results']) if isinstance(race['prediction_results'],
                                                                                          str) else race[
                    'prediction_results']

                # Parse actual results (hyphen-separated string: "1-4-2-3")
                actual_results = race['actual_results']
                if not actual_results or actual_results == 'pending':
                    continue

                # Create position mapping from actual results
                finishing_order = actual_results.split('-')
                actual_positions = {finishing_order[i]: i + 1 for i in range(len(finishing_order))}

                # Extract predictions (assuming they're in prediction_results['predictions'])
                predictions = prediction_results.get('predictions', [])
                if not predictions:
                    continue

                # Process each participant
                for participant in participants:
                    numero = str(participant.get('numero', ''))
                    if numero not in actual_positions:
                        continue

                    # Find matching prediction
                    pred_data = None
                    for pred in predictions:
                        if str(pred.get('numero', '')) == numero:
                            pred_data = pred
                            break

                    if not pred_data:
                        continue

                    # Create training sample
                    sample = {
                        'comp': race['comp'],
                        'jour': race['jour'],
                        'numero': numero,
                        'actual_position': actual_positions[numero],
                        'predicted_position': pred_data.get('predicted_position', 0),
                        'predicted_rank': pred_data.get('predicted_rank', 0),
                        # Add race features
                        'typec': race.get('typec', ''),
                        'dist': race.get('dist', 0),
                        'partant': race.get('partant', 0),
                        'hippo': race.get('hippo', ''),
                        # Add participant features (these should match training features)
                        **{k: v for k, v in participant.items() if k not in ['numero']}
                    }

                    training_samples.append(sample)

            except Exception as e:
                if self.verbose:
                    print(f"Error processing race {race.get('comp', 'unknown')}: {e}")
                continue

        training_df = pd.DataFrame(training_samples)

        if self.verbose:
            print(f"Extracted {len(training_df)} training samples from {len(races)} races")

        return training_df

    def analyze_model_performance(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current model performance on new data.

        Args:
            training_data: DataFrame with predictions and actual results

        Returns:
            Dictionary with performance analysis
        """
        if training_data.empty or self.rf_model is None:
            return {"status": "error", "message": "No data or model available"}

        if self.verbose:
            print("Analyzing current model performance...")

        # Calculate prediction errors
        predictions = training_data['predicted_position'].values
        actuals = training_data['actual_position'].values

        # Basic metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mean_error = np.mean(predictions - actuals)

        # Error analysis by position
        position_analysis = {}
        for pos in sorted(training_data['actual_position'].unique()):
            mask = training_data['actual_position'] == pos
            if mask.sum() > 0:
                pos_predictions = predictions[mask]
                pos_actuals = actuals[mask]
                position_analysis[str(pos)] = {  # Convert key to string
                    'count': int(mask.sum()),
                    'mean_error': float(np.mean(pos_predictions - pos_actuals)),
                    'mae': float(mean_absolute_error(pos_actuals, pos_predictions)),
                    'rmse': float(np.sqrt(mean_squared_error(pos_actuals, pos_predictions)))
                }

        analysis_results = {
            'sample_size': int(len(training_data)),
            'overall_mae': float(mae),
            'overall_rmse': float(rmse),
            'mean_error': float(mean_error),
            'position_analysis': position_analysis
        }

        # Save analysis
        with open(self.output_dir / 'performance_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=4)

        if self.verbose:
            print(f"Current model performance: MAE={mae:.4f}, RMSE={rmse:.4f}")

        return analysis_results

    def train_incremental_model(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train an incremental model using new data.

        Args:
            training_data: DataFrame with new training samples

        Returns:
            Dictionary with training results and model improvement metrics
        """
        if training_data.empty:
            return {"status": "error", "message": "No training data available"}

        if self.verbose:
            print("Training incremental model...")

        # Prepare features (simplified - use available numeric features)
        feature_cols = [col for col in training_data.columns
                        if
                        col not in ['comp', 'jour', 'numero', 'actual_position', 'predicted_position', 'predicted_rank']
                        and training_data[col].dtype in ['int64', 'float64']]

        X = training_data[feature_cols].fillna(0)
        y = training_data['actual_position']

        if len(X.columns) == 0:
            return {"status": "error", "message": "No usable features found"}

        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Evaluate baseline (current model performance)
        if self.rf_model is not None:
            try:
                # For baseline, use predicted positions as "predictions"
                baseline_predictions = training_data.loc[X_test.index, 'predicted_position']
                baseline_mae = mean_absolute_error(y_test, baseline_predictions)
                baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
            except:
                baseline_mae = float('inf')
                baseline_rmse = float('inf')
        else:
            baseline_mae = float('inf')
            baseline_rmse = float('inf')

        # Train new incremental model
        incremental_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        incremental_model.fit(X_train, y_train)

        # Evaluate new model
        new_predictions = incremental_model.predict(X_test)
        new_mae = mean_absolute_error(y_test, new_predictions)
        new_rmse = np.sqrt(mean_squared_error(y_test, new_predictions))

        # Calculate improvement
        mae_improvement = ((baseline_mae - new_mae) / baseline_mae) * 100 if baseline_mae != float('inf') else 0
        rmse_improvement = ((baseline_rmse - new_rmse) / baseline_rmse) * 100 if baseline_rmse != float('inf') else 0

        results = {
            "status": "success",
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features_used": int(len(X.columns)),
            "baseline_performance": {
                "mae": float(baseline_mae) if baseline_mae != float('inf') else None,
                "rmse": float(baseline_rmse) if baseline_rmse != float('inf') else None
            },
            "new_model_performance": {
                "mae": float(new_mae),
                "rmse": float(new_rmse)
            },
            "improvement": {
                "mae_improvement_pct": float(mae_improvement),
                "rmse_improvement_pct": float(rmse_improvement),
                "significant": bool(mae_improvement > self.improvement_threshold * 100)
            },
            "model": incremental_model,
            "feature_columns": list(X.columns)
        }

        if self.verbose:
            print(f"Incremental model training complete:")
            print(f"  MAE: {baseline_mae:.4f} -> {new_mae:.4f} ({mae_improvement:+.2f}%)")
            print(f"  RMSE: {baseline_rmse:.4f} -> {new_rmse:.4f} ({rmse_improvement:+.2f}%)")
            print(f"  Significant improvement: {results['improvement']['significant']}")

        return results

    def save_incremental_model(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save the incremental model if it shows significant improvement.

        Args:
            training_results: Results from incremental training

        Returns:
            Dictionary with save results
        """
        if not training_results.get('improvement', {}).get('significant', False):
            return {
                "status": "skipped",
                "reason": "No significant improvement found"
            }

        if self.verbose:
            print("Saving incremental model...")

        # Create version string
        db_type = self.db_name
        version = f"{db_type}_incremental_v{datetime.now().strftime('%Y%m%d_%H%M')}"

        # Prepare model for saving
        incremental_model = training_results['model']

        # Use the standard model manager save method
        saved_paths = self.model_manager.save_models(
            rf_model=incremental_model,
            lstm_model=lstm_model,  # Don't retrain LSTM incrementally for now
            feature_state={
                'preprocessing_params': self.orchestrator.preprocessing_params,
                'embedding_dim': self.orchestrator.embedding_dim,
                'feature_columns': training_results['feature_columns'],
                'incremental_training': True
            }
        )

        # Save training metadata
        metadata = {
            'version': version,
            'model_type': 'incremental_rf',
            'base_model': str(self.model_path),
            'training_date': datetime.now().isoformat(),
            'performance_improvement': training_results['improvement'],
            'training_summary': {
                'samples': training_results['training_samples'],
                'features': training_results['features_used']
            }
        }

        metadata_path = self.output_dir / 'incremental_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        if self.verbose:
            print(f"Incremental model saved: {version}")

        return {
            "status": "success",
            "version": version,
            "model_paths": saved_paths,
            "metadata_path": str(metadata_path),
            "improvement": training_results['improvement']
        }

    def archive_processed_races(self, races: List[Dict]) -> Dict[str, Any]:
        """
        Archive processed races from daily_race to historical_races.

        Args:
            races: List of race dictionaries that were successfully processed

        Returns:
            Dictionary with archiving results
        """
        if not races:
            return {"status": "skipped", "reason": "No races to archive"}

        if self.verbose:
            print(f"Archiving {len(races)} processed races...")

        try:
            # Use the race archiver to move races
            archive_results = self.race_archiver.archive_races(
                race_comps=[race['comp'] for race in races],
                validate_before_archive=True
            )

            if self.verbose:
                print(f"Archived {archive_results.get('archived_count', 0)} races")

            return archive_results

        except Exception as e:
            if self.verbose:
                print(f"Error archiving races: {e}")
            return {"status": "error", "message": str(e)}

    def run_incremental_training_pipeline(self, date_from: str = None, date_to: str = None,
                                          limit: int = None) -> Dict[str, Any]:
        """
        Run the complete incremental training pipeline for both RF and LSTM models.

        Args:
            date_from: Start date for processing races
            date_to: End date for processing races
            limit: Maximum number of races to process

        Returns:
            Dictionary with complete pipeline results
        """
        start_time = datetime.now()

        if self.verbose:
            print(f"Starting incremental training pipeline: {date_from} to {date_to}")

        pipeline_results = {
            "status": "success",
            "execution_time": 0.0,
            "races_fetched": 0,
            "training_data_extracted": 0,
            "performance_analysis": {},
            "rf_training": {},
            "lstm_training": {},
            "model_saved": {},
            "races_archived": {}
        }

        try:
            # Step 1: Fetch completed races from daily_race table
            races = self.fetch_completed_races(date_from, date_to, limit)
            pipeline_results["races_fetched"] = int(len(races))

            if not races:
                pipeline_results["status"] = "warning"
                pipeline_results["message"] = "No completed races found"
                return pipeline_results

            # Step 2: Extract RF training data (existing method)
            rf_data = self.extract_training_data(races)
            pipeline_results["training_data_extracted"] = int(len(rf_data))

            if rf_data.empty:
                pipeline_results["status"] = "warning"
                pipeline_results["message"] = "No RF training data extracted"
                return pipeline_results

            # Step 3: Extract LSTM training data (new method)
            lstm_data = self.extract_lstm_training_data(races)

            if lstm_data:
                pipeline_results["lstm_data_extracted"] = int(lstm_data.get('sample_count', 0))
                if self.verbose:
                    print(f"LSTM data extracted: {lstm_data.get('sample_count', 0)} samples")
            else:
                pipeline_results["lstm_data_extracted"] = 0
                if self.verbose:
                    print("No LSTM training data extracted")

            # Step 4: Analyze current model performance
            performance_analysis = self.analyze_model_performance(rf_data)
            pipeline_results["performance_analysis"] = performance_analysis

            # Step 5: Train incremental RF model
            rf_training_results = self.train_incremental_model(rf_data)
            pipeline_results["rf_training"] = rf_training_results

            # Step 6: Train incremental LSTM model if data available
            if lstm_data and 'X_sequences' in lstm_data:
                lstm_training_results = self.train_incremental_lstm(lstm_data)
                pipeline_results["lstm_training"] = lstm_training_results
            else:
                pipeline_results["lstm_training"] = {
                    "status": "skipped",
                    "message": "No LSTM data available"
                }

            if self.verbose:
                print("pipeline STEP 7 NOW")

            # Step 7: Determine if we should save models
            rf_significant = rf_training_results.get("improvement", {}).get("significant", False)
            lstm_significant = pipeline_results["lstm_training"].get("improvement", {}).get("significant", False)
            overall_significant = rf_significant or lstm_significant

            if self.verbose:
                print(f"Model improvements: RF={rf_significant}, LSTM={lstm_significant}")

            # Step 8: Save models if significant improvement found (or fallback to base models)
            if overall_significant:
                # Get models to save - use improved models if available, otherwise use base models
                new_rf_model = rf_training_results.get("model") if rf_significant else self.rf_model

                # For LSTM: use new model if improved, otherwise use base model as fallback
                if lstm_significant and pipeline_results["lstm_training"].get("model"):
                    new_lstm_model = pipeline_results["lstm_training"]["model"]
                    lstm_source = "retrained"
                else:
                    new_lstm_model = self.lstm_model  # Use base model as fallback
                    lstm_source = "base_model_copy"

                # Prepare feature state
                feature_state = {
                    'preprocessing_params': self.orchestrator.preprocessing_params,
                    'embedding_dim': self.orchestrator.embedding_dim,
                    'sequence_length': getattr(self.orchestrator, 'sequence_length', 5),
                    'incremental_training': True,
                    'training_date': datetime.now().isoformat(),
                    'lstm_source': lstm_source  # Track whether LSTM was retrained or copied
                }

                # Add RF feature columns if available
                if rf_training_results.get("feature_columns"):
                    feature_state['rf_feature_columns'] = rf_training_results["feature_columns"]

                # Save models using model manager
                try:
                    saved_paths = self.model_manager.save_models(
                        rf_model=new_rf_model,
                        lstm_model=new_lstm_model,
                        feature_state=feature_state
                    )

                    # Create version string for metadata
                    db_type = self.db_name
                    version = f"{db_type}_incremental_v{datetime.now().strftime('%Y%m%d_%H%M')}"

                    # Save comprehensive training metadata
                    metadata = {
                        'version': version,
                        'model_type': 'hybrid_incremental',
                        'base_model': str(self.model_path),
                        'training_date': datetime.now().isoformat(),
                        'models_trained': {
                            'rf': rf_significant,
                            'lstm': lstm_significant
                        },
                        'model_sources': {
                            'rf': 'retrained' if rf_significant else 'base_model_copy',
                            'lstm': lstm_source
                        },
                        'performance_improvements': {
                            'rf': rf_training_results.get("improvement", {}),
                            'lstm': pipeline_results["lstm_training"].get("improvement", {})
                        },
                        'training_summary': {
                            'rf_samples': rf_training_results.get("training_samples", 0),
                            'rf_features': rf_training_results.get("features_used", 0),
                            'lstm_samples': pipeline_results["lstm_training"].get("training_samples", 0),
                            'lstm_sequence_length': pipeline_results["lstm_training"].get("sequence_length", 0),
                            'lstm_sequential_features': pipeline_results["lstm_training"].get("sequential_features", 0),
                            'lstm_static_features': pipeline_results["lstm_training"].get("static_features", 0)
                        }
                    }

                    metadata_path = self.output_dir / 'incremental_model_metadata.json'
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)

                    pipeline_results["model_saved"] = {
                        "status": "success",
                        "version": version,
                        "model_paths": saved_paths,
                        "metadata_path": str(metadata_path),
                        "models_updated": {
                            "rf": rf_significant,
                            "lstm": lstm_significant
                        },
                        "model_sources": {
                            "rf": "retrained" if rf_significant else "base_model_copy",
                            "lstm": lstm_source
                        }
                    }

                    if self.verbose:
                        print(f"Incremental models saved: {version}")
                        if rf_significant:
                            print("  ✓ RF model retrained")
                        else:
                            print("  → RF model copied from base")

                        if lstm_significant:
                            print("  ✓ LSTM model retrained")
                        else:
                            print("  → LSTM model copied from base (fallback)")

                except Exception as e:
                    pipeline_results["model_saved"] = {
                        "status": "error",
                        "message": f"Error saving models: {str(e)}"
                    }
                    if self.verbose:
                        print(f"Error saving models: {e}")

            else:
                pipeline_results["model_saved"] = {
                    "status": "skipped",
                    "reason": "No significant improvement found in either model"
                }
                if self.verbose:
                    print("No significant improvements found - models not saved")

            # Step 9: Archive races only if model was saved successfully
            if pipeline_results["model_saved"].get("status") == "success":
                try:
                    # Archive races to historical_races
                    archive_results = self.race_archiver.archive_races(races)
                    pipeline_results["races_archived"] = archive_results

                    # If archiving was successful, immediately clean up using archiver's method
                    if archive_results.get('successful', 0) > 0:
                        # Clean up races with 0 days (immediate cleanup)
                        cleanup_results = self.race_archiver.clean_archived_races(
                            older_than_days=0,  # Immediate cleanup
                            dry_run=False  # Actually delete
                        )
                        pipeline_results["races_cleaned"] = cleanup_results

                        if self.verbose:
                            cleaned_count = cleanup_results.get('count', 0)
                            print(f"Cleaned up {cleaned_count} archived races from daily_race table")

                    self.processed_races = races

                except Exception as e:
                    pipeline_results["races_archived"] = {
                        "status": "error",
                        "message": f"Error archiving races: {str(e)}"
                    }
                    if self.verbose:
                        print(f"Error archiving races: {e}")
            else:
                pipeline_results["races_archived"] = {
                    "status": "skipped",
                    "reason": "Model was not saved successfully"
                }

        except Exception as e:
            pipeline_results["status"] = "error"
            pipeline_results["message"] = str(e)
            if self.verbose:
                print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()

        # Calculate execution time
        pipeline_results["execution_time"] = (datetime.now() - start_time).total_seconds()

        # Save complete results with proper JSON serialization
        def json_serialize(obj):
            """Custom JSON serializer for numpy/pandas types."""
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj)

        # Create a clean copy for JSON serialization (remove model objects)
        json_results = pipeline_results.copy()
        if 'model' in json_results.get("rf_training", {}):
            json_results["rf_training"] = {k: v for k, v in json_results["rf_training"].items() if k != 'model'}
        if 'model' in json_results.get("lstm_training", {}):
            json_results["lstm_training"] = {k: v for k, v in json_results["lstm_training"].items() if k != 'model'}

        with open(self.output_dir / 'incremental_training_results.json', 'w') as f:
            json.dump(json_results, f, indent=4, default=json_serialize)

        if self.verbose:
            print(f"Incremental training pipeline completed in {pipeline_results['execution_time']:.2f} seconds")

        return pipeline_results

    def convert_daily_to_historical_format(self, races: List[Dict]) -> pd.DataFrame:
        """
        Convert daily race data to historical race format for orchestrator processing.

        Args:
            races: List of race dictionaries from daily_race table

        Returns:
            DataFrame in historical_races format
        """
        if self.verbose:
            print("Converting daily races to historical format...")

        historical_rows = []

        for race in races:
            try:
                # Parse JSON fields
                participants = json.loads(race['participants']) if isinstance(race['participants'], str) else race[
                    'participants']

                # Parse actual results to get final positions
                actual_results = race['actual_results']
                if not actual_results or actual_results == 'pending':
                    continue

                finishing_order = actual_results.split('-')
                actual_positions = {finishing_order[i]: i + 1 for i in range(len(finishing_order))}

                # Add final_position to each participant
                for participant in participants:
                    numero = str(participant.get('numero', ''))
                    if numero in actual_positions:
                        participant['final_position'] = actual_positions[numero]
                        participant['cl'] = actual_positions[numero]  # Alternative name

                # Create historical race row
                historical_row = {
                    'comp': race['comp'],
                    'jour': race['jour'],
                    'reunion': race.get('reun', ''),
                    'prix': race.get('prix', ''),
                    'quinte': race.get('quinte', False),
                    'hippo': race.get('hippo', ''),
                    'meteo': race.get('meteo', ''),
                    'dist': race.get('dist', 0),
                    'corde': race.get('corde', ''),
                    'natpis': race.get('natpis', ''),
                    'pistegp': race.get('pistegp', ''),
                    'typec': race.get('typec', ''),
                    'partant': race.get('partant', 0),
                    'temperature': race.get('temperature', 0),
                    'forceVent': race.get('forceVent', 0),
                    'directionVent': race.get('directionVent', ''),
                    'nebulosite': race.get('nebulosite', ''),
                    'participants': json.dumps(participants),
                    'created_at': race.get('created_at', datetime.now().isoformat())
                }

                historical_rows.append(historical_row)

            except Exception as e:
                if self.verbose:
                    print(f"Error converting race {race.get('comp', 'unknown')}: {e}")
                continue

        historical_df = pd.DataFrame(historical_rows)

        if self.verbose:
            print(f"Converted {len(historical_df)} races to historical format")

        return historical_df

    def extract_lstm_training_data(self, races: List[Dict]) -> Dict[str, Any]:
        if self.verbose:
            print("Extracting LSTM training data using orchestrator pipeline...")

        # Convert daily races to historical format
        daily_historical = self.convert_daily_to_historical_format(races)

        if daily_historical.empty:
            return {}

        # Get unique horse IDs from daily races (batch collection)
        daily_horses = set()
        for race in races:
            try:
                participants = json.loads(race['participants']) if isinstance(race['participants'], str) else race[
                    'participants']
                for participant in participants:
                    if 'idche' in participant and participant['idche']:
                        daily_horses.add(str(participant['idche']))
            except Exception as e:
                if self.verbose:
                    print(f"Error extracting horse IDs from race {race.get('comp', 'unknown')}: {e}")
                continue

        if not daily_horses:
            if self.verbose:
                print("No horse IDs found in daily races")
            return {}

        if self.verbose:
            print(f"Found {len(daily_horses)} unique horses in daily races")

        # Load historical data for ALL horses in one query
        try:
            historical_data = self.get_races_by_horses(list(daily_horses), limit_per_horse=50)

            if self.verbose:
                if not historical_data.empty:
                    horses_found = historical_data['horse_id'].nunique()
                    total_races = len(historical_data)
                    avg_races = total_races / horses_found if horses_found > 0 else 0
                    print(
                        f"Loaded {total_races} historical races for {horses_found} horses (avg {avg_races:.1f} races per horse)")
                else:
                    print("No historical data found for daily race horses")

        except Exception as e:
            if self.verbose:
                print(f"Error loading historical races: {e}")
            return {}

        if historical_data.empty:
            if self.verbose:
                print("No historical data found for daily race horses")
            return {}

        # Remove the horse_id column and duplicates
        historical_data = historical_data.drop(columns=['horse_id'], errors='ignore')
        historical_data = historical_data.drop_duplicates(subset=['comp'])

        # Combine with daily races
        combined_df = pd.concat([historical_data, daily_historical], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['comp'])
        combined_df = combined_df.sort_values('jour')

        if self.verbose:
            print(
                f"Combined dataset: {len(historical_data)} historical + {len(daily_historical)} daily = {len(combined_df)} total races")

        # Rest of your existing orchestrator processing...
        try:
            expanded_df = self.orchestrator._expand_participants(combined_df)
            complete_df = self.orchestrator.prepare_complete_dataset(expanded_df, use_cache=False)
            X_sequences, X_static, y_lstm = self.orchestrator.extract_lstm_features(complete_df)

            lstm_data = {
                'X_sequences': X_sequences,
                'X_static': X_static,
                'y': y_lstm,
                'complete_df': complete_df,
                'sample_count': len(y_lstm),
                'sequence_length': X_sequences.shape[1],
                'sequential_features': X_sequences.shape[2],
                'static_features': X_static.shape[1]
            }

            if self.verbose:
                print(f"Extracted LSTM data: {len(y_lstm)} samples")
                print(f"  Sequence length: {X_sequences.shape[1]}")
                print(f"  Sequential features: {X_sequences.shape[2]}")
                print(f"  Static features: {X_static.shape[1]}")

            return lstm_data

        except Exception as e:
            if self.verbose:
                print(f"Error in LSTM data extraction: {e}")
            return {}

    def get_races_by_horses(self, idche_list: List[str], limit_per_horse: int = 50) -> pd.DataFrame:
        """Get historical races for multiple horses in a single query."""
        if not idche_list:
            return pd.DataFrame()

        conn = sqlite3.connect(self.db_path)

        # Create placeholders for the IN clause
        placeholders = ','.join(['?' for _ in idche_list])

        query = f"""
        SELECT hr.*, 
               json_extract(participant.value, '$.idche') as horse_id
        FROM historical_races hr,
             json_each(hr.participants) as participant
        WHERE json_extract(participant.value, '$.idche') IN ({placeholders})
        ORDER BY json_extract(participant.value, '$.idche'), hr.jour DESC
        """

        # Convert idche_list to ensure all are strings
        params = [int(idche) for idche in idche_list]

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Limit races per horse if needed
        if limit_per_horse:
            df = df.groupby('horse_id').head(limit_per_horse)

        return df
    def train_incremental_lstm(self, lstm_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train an incremental LSTM model using new data.

        Args:
            lstm_data: Dictionary with LSTM features and targets

        Returns:
            Dictionary with LSTM training results
        """
        print("I'm in train_incremental ldstm")
        if not lstm_data or 'X_sequences' not in lstm_data:
            print("Error with IF #1")
            return {"status": "error", "message": "No LSTM training data available"}

        if self.lstm_model is None:
            print("Error with IF #2")
            return {"status": "error", "message": "No base LSTM model available"}

        if self.verbose:
            print("Training incremental LSTM model...")



        try:
            X_sequences = lstm_data['X_sequences']
            X_static = lstm_data['X_static']
            y_lstm = lstm_data['y']

            # Split LSTM data for training/testing
            from sklearn.model_selection import train_test_split
            X_seq_train, X_seq_test, X_static_train, X_static_test, y_lstm_train, y_lstm_test = train_test_split(
                X_sequences, X_static, y_lstm, test_size=0.3, random_state=42
            )

            # Evaluate baseline LSTM performance
            baseline_lstm_pred = self.lstm_model.predict([X_seq_test, X_static_test], verbose=0).flatten()
            baseline_lstm_mae = mean_absolute_error(y_lstm_test, baseline_lstm_pred)
            baseline_lstm_rmse = np.sqrt(mean_squared_error(y_lstm_test, baseline_lstm_pred))

            # Create and train new LSTM model
            lstm_model_components = self.orchestrator.create_hybrid_model(
                sequence_shape=X_seq_train.shape,
                static_shape=X_static_train.shape,
                lstm_units=64,
                dropout_rate=0.2
            )

            if lstm_model_components is None:
                return {"status": "error", "message": "LSTM model creation failed"}

            new_lstm_model = lstm_model_components['lstm']

            # Train with early stopping
            from tensorflow.keras.callbacks import EarlyStopping
            callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

            if self.verbose:
                print("Training LSTM model...")

            history = new_lstm_model.fit(
                [X_seq_train, X_static_train], y_lstm_train,
                validation_data=([X_seq_test, X_static_test], y_lstm_test),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1 if self.verbose else 0
            )

            # Evaluate new LSTM model
            new_lstm_pred = new_lstm_model.predict([X_seq_test, X_static_test], verbose=0).flatten()
            new_lstm_mae = mean_absolute_error(y_lstm_test, new_lstm_pred)
            new_lstm_rmse = np.sqrt(mean_squared_error(y_lstm_test, new_lstm_pred))

            # Calculate improvement
            lstm_mae_improvement = ((baseline_lstm_mae - new_lstm_mae) / baseline_lstm_mae) * 100
            lstm_rmse_improvement = ((baseline_lstm_rmse - new_lstm_rmse) / baseline_lstm_rmse) * 100

            results = {
                "status": "success",
                "training_samples": int(len(X_seq_train)),
                "test_samples": int(len(X_seq_test)),
                "sequence_length": int(X_seq_train.shape[1]),
                "sequential_features": int(X_seq_train.shape[2]),
                "static_features": int(X_static_train.shape[1]),
                "baseline_performance": {
                    "mae": float(baseline_lstm_mae),
                    "rmse": float(baseline_lstm_rmse)
                },
                "new_performance": {
                    "mae": float(new_lstm_mae),
                    "rmse": float(new_lstm_rmse)
                },
                "improvement": {
                    "mae_improvement_pct": float(lstm_mae_improvement),
                    "rmse_improvement_pct": float(lstm_rmse_improvement),
                    "significant": bool(lstm_mae_improvement > 5.0)  # 5% threshold
                },
                "model": new_lstm_model,
                "training_history": {
                    "epochs": len(history.history['loss']),
                    "final_loss": float(history.history['loss'][-1]),
                    "final_val_loss": float(history.history['val_loss'][-1])
                }
            }

            if self.verbose:
                print(f"LSTM training complete:")
                print(f"  MAE LSTM: {baseline_lstm_mae:.4f} -> {new_lstm_mae:.4f} ({lstm_mae_improvement:+.2f}%)")
                print(f"  RMSE LSTM: {baseline_lstm_rmse:.4f} -> {new_lstm_rmse:.4f} ({lstm_rmse_improvement:+.2f}%)")
                print(f"  Significant improvement: {results['improvement']['significant']}")

            return results

        except Exception as e:
            if self.verbose:
                print(f"Error training LSTM model: {e}")
            return {"status": "error", "message": str(e)}

def main():
    """
    Main function for IDE usage - Incremental Training Pipeline.
    """
    # Configuration
    DB_NAME = None  # Use active_db from config
    OUTPUT_DIR = None  # Auto-generated timestamp directory

    # Date range for processing (modify as needed)
    FROM_DATE = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')  # Last 7 days
    TO_DATE = datetime.now().strftime('%Y-%m-%d')  # Today
    LIMIT = None  # Process all available races

    VERBOSE = True

    # Create pipeline
    pipeline = IncrementalTrainingPipeline(
        model_path=None,  # Use latest model from config
        db_name=DB_NAME,
        output_dir=OUTPUT_DIR,
        verbose=VERBOSE
    )

    # Run incremental training pipeline
    results = pipeline.run_incremental_training_pipeline(
        date_from=FROM_DATE,
        date_to=TO_DATE,
        limit=LIMIT
    )


    # Print summary results
    print("\n" + "=" * 60)
    print("INCREMENTAL TRAINING PIPELINE RESULTS")
    print("=" * 60)

    if results["status"] == "success":
        print(f"Races fetched: {results['races_fetched']}")
        print(f"Training samples: {results['training_data_extracted']}")

        if 'performance_analysis' in results:
            perf = results['performance_analysis']
            print(f"\nCurrent Model Performance:")
            print(f"  MAE: {perf.get('overall_mae', 'N/A'):.4f}")
            print(f"  RMSE: {perf.get('overall_rmse', 'N/A'):.4f}")

        if results['incremental_training'].get('status') == 'success':
            training = results['incremental_training']
            improvement = training['improvement']
            print(f"\nIncremental Training:")
            print(f"  MAE improvement: {improvement['mae_improvement_pct']:+.2f}%")
            print(f"  RMSE improvement: {improvement['rmse_improvement_pct']:+.2f}%")
            print(f"  Significant: {improvement['significant']}")

        if results['model_saved'].get('status') == 'success':
            print(f"\nModel Saved: {results['model_saved']['version']}")

        if results['races_archived'].get('status') == 'success':
            archived = results['races_archived']
            print(f"\nRaces Archived: {archived.get('archived_count', 0)}")

        print(f"\nExecution time: {results['execution_time']:.2f} seconds")
        print(f"Results saved to: {pipeline.output_dir}")

    else:
        print(f"Pipeline failed: {results.get('message', 'Unknown error')}")

    return results


if __name__ == "__main__":
    main()