
import yaml
import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import threading
import queue
import sys
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent  # UI directory
project_root = current_dir.parent    # HorseAIv2 directory
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.env_setup import AppConfig
from model_training.historical import train_race_model
from core.connectors.api_daily_sync import RaceFetcher
from core.orchestrators.prediction_orchestrator import PredictionOrchestrator
from utils.predict_evaluator import PredictEvaluator
from model_training.regressions.regression_enhancement import IncrementalTrainingPipeline
from utils.ai_advisor import BettingAdvisor
from race_prediction.race_predict import (
    re_blend_predictions_with_dynamic_weights as reblend_predictions_func,
    predict_races_fast
)

class PipelineHelper:
    """Helper class for config management and status calculations"""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self._config_data = self.load_config()

        # Training management
        self.training_queue = queue.Queue()
        self.training_thread = None
        self.is_training = False

    def load_config(self) -> Optional[Dict[str, Any]]:
        """Load config.yaml file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    self._config_data = yaml.safe_load(file)
                    return self._config_data
            return None
        except Exception as e:
            raise Exception(f"Error loading config: {str(e)}")

    def save_config(self, updated_config: Dict[str, Any]) -> bool:
        """Save updated config to config.yaml"""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(updated_config, file, default_flow_style=False, sort_keys=False)
            self._config_data = updated_config
            return True
        except Exception as e:
            raise Exception(f"Error saving config: {str(e)}")

    def get_config_json(self) -> str:
        """Get current config as formatted JSON"""
        if self._config_data:
            return json.dumps(self._config_data, indent=2)
        return "No configuration loaded"

    def get_active_db(self) -> str:
        """Get active database from config"""
        if self._config_data and 'base' in self._config_data:
            return self._config_data['base'].get('active_db', 'Unknown')
        return 'Unknown'

    def get_sqlite_databases(self) -> list:
        """Get list of SQLite databases from config"""
        if not self._config_data or 'databases' not in self._config_data:
            return []
        return [db['name'] for db in self._config_data['databases'] if db.get('type') == 'sqlite']

    def get_last_training_info(self) -> Tuple[str, str]:
        """Extract last training date and version from model names"""
        if not self._config_data or 'models' not in self._config_data:
            return 'Unknown', 'Unknown'

        models = self._config_data['models']
        latest_models = models.get('latest_models', {})

        if latest_models:
            # Get the most recent model (try RF first, then TabNet)
            model_path = latest_models.get('rf') or latest_models.get('tabnet')

            if model_path:
                # Extract date from model path like "2025-09-22/2years_084846"
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', model_path)
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        training_date = datetime.strptime(date_str, '%Y-%m-%d')
                        days_ago = (datetime.now() - training_date).days

                        if days_ago == 0:
                            last_training = "Today"
                        elif days_ago == 1:
                            last_training = "Yesterday"
                        else:
                            last_training = f"{days_ago} days ago"

                        # Extract version from path (everything after the date)
                        version_part = model_path.split('/')[-1] if '/' in model_path else model_path
                        return last_training, version_part
                    except:
                        pass

        return 'Unknown', 'Unknown'

    def get_model_paths(self) -> Dict[str, str]:
        """Get the current RF and TabNet model paths from config"""
        if not self._config_data or 'models' not in self._config_data:
            return {}

        models = self._config_data['models']
        latest_models = models.get('latest_models', {})
        model_dir = models.get('model_dir', './models')

        # Build full paths
        model_paths = {}
        if 'rf' in latest_models:
            model_paths['rf'] = f"{model_dir}/{latest_models['rf']}"
        if 'tabnet' in latest_models:
            model_paths['tabnet'] = f"{model_dir}/{latest_models['tabnet']}"

        return model_paths

    def get_model_file_paths(self) -> Dict[str, str]:
        """Get specific model file paths for RF and TabNet models"""
        model_paths = self.get_model_paths()

        model_files = {}

        # RF model file (joblib format) - no hybrid prefix
        if 'rf' in model_paths:
            model_files['rf'] = f"{model_paths['rf']}/rf_model.joblib"

        # TabNet model file (zip format) - no hybrid prefix
        if 'tabnet' in model_paths:
            model_files['tabnet'] = f"{model_paths['tabnet']}/tabnet_model.zip"

        return model_files

    def get_system_status(self) -> Dict[str, str]:
        """Get complete system status"""
        if not self._config_data:
            self.load_config()

        last_training, model_version = self.get_last_training_info()

        return {
            'active_db': self.get_active_db(),
            'last_training': last_training,
            'model_version': model_version
        }

    def get_config_sections(self) -> Dict[str, Any]:
        """Get organized config sections for UI editing"""
        if not self._config_data:
            return {}

        return {
            'base': self._config_data.get('base', {}),
            'features': self._config_data.get('features', {}),
            'tabnet': self._config_data.get('tabnet', {}),
            'dataset': self._config_data.get('dataset', {}),
            'cache': self._config_data.get('cache', {}),
            'databases': self._config_data.get('databases', []),
            'blend': self._config_data.get('blend', {})
        }

    def update_config_section(self, section: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a specific config section and return full config"""
        if not self._config_data:
            self.load_config()

        config_copy = self._config_data.copy()

        if section in config_copy:
            config_copy[section].update(updates)
        else:
            config_copy[section] = updates

        return config_copy

    def execute_full_training(self, progress_callback) -> Dict[str, Any]:
        """Execute full model training"""
        try:
            # Simply call the main training function
            train_race_model.main(progress_callback)

            return {
                "success": True,
                "message": "Training completed successfully"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Training failed: {str(e)}"
            }

    def start_training_async(self) -> bool:
        """Start training in background thread"""
        if self.is_training:
            return False

        self.is_training = True

        # Create background thread with name for tracking
        self.training_thread = threading.Thread(
            target=self._training_worker,
            daemon=True,
            name="TrainingWorkerThread"
        )
        self.training_thread.start()

        # Send initial status with thread info
        import os
        self.training_queue.put({
            'type': 'info',
            'message': f'Training started in background',
            'process_id': os.getpid(),
            'thread_id': self.training_thread.ident,
            'thread_name': self.training_thread.name,
            'is_alive': self.training_thread.is_alive()
        })

        return True

    def _training_worker(self):
        """Background training worker"""
        import os
        import threading
        from datetime import datetime

        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        start_time = datetime.now()

        # Send worker started message
        self.training_queue.put({
            'type': 'worker_started',
            'message': f'Training worker thread started',
            'thread_id': thread_id,
            'thread_name': thread_name,
            'process_id': os.getpid(),
            'start_time': start_time.isoformat()
        })

        try:
            def progress_callback(percent, message):
                # Enhanced progress callback with thread info
                self.training_queue.put({
                    'type': 'progress',
                    'percent': percent,
                    'message': message,
                    'thread_id': thread_id,
                    'timestamp': datetime.now().isoformat()
                })

            # Run actual training
            train_race_model.main(progress_callback=progress_callback)

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Send completion
            self.training_queue.put({
                'type': 'complete',
                'success': True,
                'message': 'Training completed successfully',
                'duration_seconds': duration,
                'thread_id': thread_id
            })
        except Exception as e:
            import traceback
            duration = (datetime.now() - start_time).total_seconds()

            self.training_queue.put({
                'type': 'complete',
                'success': False,
                'error': str(e),
                'message': f'Training failed: {str(e)}',
                'traceback': traceback.format_exc(),
                'duration_seconds': duration,
                'thread_id': thread_id
            })
        finally:
            self.is_training = False
            self.training_queue.put({
                'type': 'worker_stopped',
                'message': 'Training worker thread stopped',
                'thread_id': thread_id
            })

    def get_training_updates(self) -> List[Dict[str, Any]]:
        """Get latest training updates"""
        updates = []
        try:
            while True:
                update = self.training_queue.get_nowait()
                updates.append(update)
        except queue.Empty:
            pass
        return updates

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training thread status"""
        import os

        status = {
            'is_training': self.is_training,
            'process_id': os.getpid(),
            'thread_exists': self.training_thread is not None if hasattr(self, 'training_thread') else False
        }

        if hasattr(self, 'training_thread') and self.training_thread is not None:
            status.update({
                'thread_id': self.training_thread.ident,
                'thread_name': self.training_thread.name,
                'thread_alive': self.training_thread.is_alive(),
                'thread_daemon': self.training_thread.daemon
            })

        return status
    # Prediction operations
    def get_daily_races(self) -> List[Dict[str, Any]]:
        """Get races from daily_race table for a specific date"""
        if not self._config_data:
          self.load_config()

            # Initialize race fetcher
        race_fetcher = RaceFetcher()

            # Get races for the date
        races = race_fetcher.get_all_daily_races()

        return races

    def sync_daily_races(self,date) -> List[Dict[str, Any]]:
        race_fetcher = RaceFetcher()
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")
        race_fetcher.fetch_and_store_daily_races(date)

    def get_races_needing_prediction(self) -> List[Dict[str, Any]]:
        """Get races that need predictions (don't have prediction_results)"""
        date = datetime.today().strftime("%Y-%m-%d")

        races = self.get_daily_races()

            # Filter races that need predictions
        unpredicted_races = [
            race for race in races
            if race.get('has_predictions', 0) == 0 and race.get('has_processed_data', 0) == 1
        ]

        return unpredicted_races

    def get_races_for_reprediction(self, date: str = None) -> List[Dict[str, Any]]:
        """Get all races with processed data (for force reprediction)"""
        try:
            races = self.get_daily_races()

            # Filter races that have processed data (ignore prediction status)
            reprediction_races = [
                race for race in races
                if race.get('has_processed_data', 0) == 1
            ]

            return reprediction_races

        except Exception as e:
            raise Exception(f"Error getting races for reprediction: {str(e)}")

    def execute_predictions(self, races_to_predict: List[str] = None, progress_callback=None,
                            force_reprediction: bool = False, use_batch_mode: bool = True,
                            n_jobs: int = -1, chunk_size: int = 50,
                            max_memory_mb: float = 4096) -> Dict[str, Any]:
        """
        Execute predictions for specified races or all unpredicted races.

        Args:
            races_to_predict: List of specific race IDs to predict
            progress_callback: Callback for progress updates
            force_reprediction: Predict all races regardless of existing predictions
            use_batch_mode: Use fast batch prediction (default: True, 8-16x faster!)
            n_jobs: Number of parallel workers (-1 = all cores, only used if use_batch_mode=True)
            chunk_size: Races per chunk for memory management (default: 50)
            max_memory_mb: Maximum memory usage in MB before cleanup (default: 4GB)
        """
        try:
            if progress_callback:
                progress_callback(5, "Getting races to predict...")

            # Determine which races to predict
            if races_to_predict:
                race_ids = races_to_predict
                message_type = "specified races"
            else:
                # Predict races based on force_reprediction flag
                if force_reprediction:
                    races_to_process = self.get_races_for_reprediction()
                    message_type = "all processed races (force reprediction)"
                else:
                    races_to_process = self.get_races_needing_prediction()
                    message_type = "unpredicted races"

                if not races_to_process:
                    if progress_callback:
                        progress_callback(100, f"No {message_type} found")
                    return {
                        "success": True,
                        "message": f"No {message_type} found",
                        "predicted_count": 0,
                        "total_races": 0
                    }

                race_ids = [race['comp'] for race in races_to_process]

            total_races = len(race_ids)

            if progress_callback:
                progress_callback(10, f"Predicting {total_races} {message_type}...")

            # NEW: Use fast batch prediction mode for 1000+ races
            if use_batch_mode:
                try:
                    # Use the new fast batch prediction with memory management!
                    results = predict_races_fast(
                        race_ids=race_ids,
                        n_jobs=n_jobs,  # Use all cores for maximum speed
                        chunk_size=chunk_size,  # Process in chunks to limit memory
                        max_memory_mb=max_memory_mb,  # Maximum memory before cleanup
                        db_name=self.get_active_db(),
                        verbose=False,  # Keep quiet for UI
                        progress_callback=progress_callback  # Pass through for dynamic updates!
                    )

                    predicted_count = len(results) if results else 0

                    return {
                        "success": True,
                        "message": f"Fast batch predictions completed: {predicted_count}/{total_races} successful (used {n_jobs if n_jobs > 0 else 'all'} CPU cores, {chunk_size} races/chunk)",
                        "predicted_count": predicted_count,
                        "total_races": total_races,
                        "mode": "batch",
                        "workers": n_jobs,
                        "chunk_size": chunk_size,
                        "max_memory_mb": max_memory_mb
                    }

                except Exception as e:
                    # Fallback to sequential mode on batch failure
                    print(f"⚠️ Batch mode failed, falling back to sequential: {e}")
                    use_batch_mode = False

            # OLD: Sequential prediction mode (slower, for compatibility)
            if not use_batch_mode:
                if progress_callback:
                    progress_callback(15, "Using sequential prediction mode...")

                # Initialize prediction orchestrator
                predictor = PredictionOrchestrator(verbose=False)

                # Get model information for diagnostics
                model_info = predictor.get_model_info()
                models_loaded = model_info.get('legacy_models', {}).get('models_loaded', {})
                model_weights = model_info.get('legacy_models', {})

                predicted_count = 0

                for i, comp in enumerate(race_ids):
                    if progress_callback:
                        progress = 15 + (i / total_races) * 80
                        progress_callback(int(progress), f"Predicting race {i+1}/{total_races}...")

                    try:
                        result = predictor.predict_race(comp)
                        if result.get('status') == 'success':
                            predicted_count += 1
                        else:
                            error_msg = result.get('message', 'Unknown error')
                            print(f"❌ PREDICTION FAILED for race {comp}: {error_msg}")
                    except Exception as e:
                        print(f"❌ EXCEPTION predicting race {comp}: {str(e)}")
                        continue

                if progress_callback:
                    progress_callback(100, f"Sequential predictions completed: {predicted_count}/{total_races} successful")

                return {
                    "success": True,
                    "message": f"Sequential predictions completed: {predicted_count}/{total_races} successful",
                    "predicted_count": predicted_count,
                    "total_races": total_races,
                    "mode": "sequential",
                    "model_info": model_info
                }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "message": f"Prediction failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
    def reblend_with_dynamic_weights(self, date: str = None, all_races: bool = True, progress_callback=None) -> Dict[str, Any]:
        """
        Re-blend existing predictions with dynamic weights without re-predicting.
        Much faster than re-running predictions.

        Args:
            date: Date to re-blend (YYYY-MM-DD). Ignored if all_races=True.
            all_races: If True, re-blends ALL races with predictions (default).
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with re-blending results
        """
        try:
            if progress_callback:
                progress_callback(10, "Initializing re-blending...")

            if progress_callback:
                if all_races:
                    progress_callback(30, "Re-calculating weights for ALL races...")
                else:
                    progress_callback(30, f"Re-calculating weights for {date}...")

            # Execute re-blending using the imported function
            result = reblend_predictions_func(
                date=date,
                all_races=all_races,
                verbose=False  # Keep it clean for UI
            )

            if progress_callback:
                progress_callback(90, "Finalizing...")

            # Format response
            races_processed = result.get('races_processed', 0)
            horses_updated = result.get('horses_updated', 0)
            races_detail = result.get('races_detail', [])

            if progress_callback:
                progress_callback(100, "Re-blending complete!")

            return {
                'success': True,
                'races_processed': races_processed,
                'horses_updated': horses_updated,
                'races_detail': races_detail,
                'message': f"Successfully re-blended {races_processed} races ({horses_updated} horses) with dynamic weights"
            }

        except Exception as e:
            import traceback
            error_msg = f"Error during re-blending: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"   Traceback: {traceback.format_exc()}")

            if progress_callback:
                progress_callback(100, f"Error: {str(e)}")

            return {
                'success': False,
                'error': str(e),
                'message': error_msg
            }



    def evaluate_all_predictions_comprehensive(self, progress_callback=None) -> Dict[str, Any]:
        """Comprehensive evaluation of all predicted races using new PredictEvaluator"""
        try:
            if progress_callback:
                progress_callback(5, "Initializing comprehensive evaluation...")

            # Initialize the new evaluator
            evaluator = PredictEvaluator()

            if progress_callback:
                progress_callback(20, "Getting all evaluable races...")

            # Get evaluation metrics
            metrics = evaluator.evaluate_all_races()

            if progress_callback:
                progress_callback(40, "Analyzing bet type performance...")

            # Get bet type wins
            bet_type_wins = evaluator.get_races_won_by_bet_type()

            if progress_callback:
                progress_callback(60, "Analyzing quinte betting strategies...")

            # Get quinte horse analysis
            quinte_analysis = evaluator.get_quinte_horse_betting_analysis()

            if progress_callback:
                progress_callback(80, "Preparing visualization data...")

            # Prepare data for charts
            chart_data = self._prepare_chart_data(metrics, bet_type_wins, quinte_analysis)

            if progress_callback:
                progress_callback(100, f"Evaluation completed for {metrics.races_evaluated} races")

            return {
                "success": True,
                "message": f"Comprehensive evaluation completed for {metrics.races_evaluated} races",
                "metrics": metrics,
                "bet_type_wins": bet_type_wins,
                "quinte_analysis": quinte_analysis,
                "chart_data": chart_data
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Comprehensive evaluation failed: {str(e)}"
            }

    def _prepare_chart_data(self, metrics, bet_type_wins, quinte_analysis):
        """Prepare data for visualization charts"""

        # Bet type performance chart data
        bet_performance = []
        for bet_type, stats in metrics.bet_win_rates.items():
            if stats['wins'] > 0:  # Only include bet types with wins
                bet_performance.append({
                    'bet_type': stats['display_name'],
                    'wins': stats['wins'],
                    'total': stats['total'],
                    'win_rate': stats['rate'] * 100,  # Convert to percentage
                    'losses': stats['total'] - stats['wins']
                })

        # Sort by win rate
        bet_performance.sort(key=lambda x: x['win_rate'], reverse=True)

        # Quinte horse strategy chart data
        quinte_strategy_data = []
        if quinte_analysis:
            scenarios = quinte_analysis['betting_scenarios']
            for scenario_name, data in scenarios.items():
                horse_count = scenario_name.replace('_horses', '').replace('_', ' ').title()
                quinte_strategy_data.append({
                    'strategy': f"{horse_count} Horses",
                    'wins': data['wins'],
                    'total': data['total'],
                    'win_rate': data['win_rate'] * 100,
                    'losses': data['total'] - data['wins']
                })

        # Overall performance summary
        overall_summary = {
            'total_races': metrics.total_races,
            'winner_accuracy': metrics.overall_winner_accuracy * 100,
            'podium_accuracy': metrics.overall_podium_accuracy * 100,
            'total_winning_bets': metrics.total_winning_bets
        }

        # Quinte performance summary
        quinte_summary = None
        if metrics.quinte_performance:
            qp = metrics.quinte_performance
            quinte_summary = {
                'total_quinte_races': qp['total_quinte_races'],
                'winner_accuracy': qp['winner_accuracy'] * 100,
                'quinte_bet_win_rate': qp['quinte_bet_win_rate'] * 100,
                'avg_podium_accuracy': qp['avg_podium_accuracy'] * 100
            }

        return {
            'bet_performance': bet_performance,
            'quinte_strategy': quinte_strategy_data,
            'overall_summary': overall_summary,
            'quinte_summary': quinte_summary
        }

    def execute_incremental_training(self, date_from: str, date_to: str, limit: int = None,
                                     update_model: bool = True, create_enhanced: bool = True,
                                     archive_after: bool = True, progress_callback=None) -> Dict[str, Any]:
        """Execute incremental training using the new IncrementalTrainingPipeline"""
        try:
            if progress_callback:
                progress_callback(5, "Initializing incremental training pipeline...")

            # Initialize the incremental training pipeline
            pipeline = IncrementalTrainingPipeline(
                model_path=None,  # Use latest model from config
                db_name=self.get_active_db(),
                verbose=True
            )

            if progress_callback:
                progress_callback(15, "Fetching completed races with predictions and results...")

            # Check how many races are available
            races = pipeline.fetch_completed_races(date_from, date_to, limit)
            if not races:
                return {
                    "success": False,
                    "message": "No completed races found with both predictions and results",
                    "training_results": {}
                }

            if progress_callback:
                progress_callback(25, f"Found {len(races)} races. Running incremental training...")

            # Run the incremental training pipeline
            results = pipeline.run_incremental_training_pipeline(
                date_from=date_from,
                date_to=date_to,
                limit=limit
            )

            if progress_callback:
                progress_callback(85, "Processing training results...")

            # Check if training was successful
            success = results.get("status") == "success"

            # Update progress based on results
            if success:
                model_saved = results.get("model_saved", {}).get("status") == "success"
                races_archived = results.get("races_archived", {}).get("status") == "success"

                if model_saved:
                    if progress_callback:
                        progress_callback(95, "Model saved successfully...")

                if races_archived:
                    if progress_callback:
                        progress_callback(98, "Races archived successfully...")

            if progress_callback:
                progress_callback(100, "Incremental training completed!")

            # Format results for UI consumption
            formatted_results = {
                "success": success,
                "message": self._format_training_message(results),
                "training_results": {
                    "performance_analysis": results.get("performance_analysis", {}),
                    "rf_training": results.get("rf_training", {}),
                    "tabnet_training": results.get("tabnet_training", {}),
                    "model_saved": results.get("model_saved", {}),
                    "races_archived": results.get("races_archived", {}),
                    "execution_time": results.get("execution_time", 0),
                    "races_processed": results.get("races_fetched", 0),
                    "training_samples": results.get("training_data_extracted", 0)
                }
            }

            return formatted_results

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Incremental training failed: {str(e)}",
                "training_results": {}
            }

    def _format_training_message(self, results: Dict[str, Any]) -> str:
        """Format a user-friendly message based on training results"""
        if results.get("status") != "success":
            return results.get("message", "Training failed")

        races_count = results.get("races_fetched", 0)
        samples_count = results.get("training_data_extracted", 0)

        message_parts = [f"Processed {races_count} races with {samples_count} training samples"]

        # RF training results
        rf_results = results.get("rf_training", {})
        if rf_results.get("status") == "success":
            rf_improvement = rf_results.get("improvement", {})
            mae_improvement = rf_improvement.get("mae_improvement_pct", 0)
            message_parts.append(f"RF model improved by {mae_improvement:.1f}%")

        # TabNet training results
        tabnet_results = results.get("tabnet_training", {})
        if tabnet_results.get("status") == "success":
            tabnet_improvement = tabnet_results.get("improvement", {})
            mae_improvement = tabnet_improvement.get("mae_improvement_pct", 0)
            message_parts.append(f"TabNet model improved by {mae_improvement:.1f}%")
        elif tabnet_results.get("status") == "skipped":
            message_parts.append("TabNet training skipped")

        # Model saving results
        model_saved = results.get("model_saved", {})
        if model_saved.get("status") == "success":
            version = model_saved.get("version", "unknown")
            message_parts.append(f"New model saved: {version}")

        # Archive results
        archived = results.get("races_archived", {})
        if archived.get("status") == "success":
            archived_count = archived.get("successful", 0)
            message_parts.append(f"Archived {archived_count} races")

        return ". ".join(message_parts)

    def execute_quinte_incremental_training(self, date_from: str, date_to: str,
                                           limit: int = None,
                                           focus_on_failures: bool = True,
                                           progress_callback=None) -> Dict[str, Any]:
        """Execute quinté incremental training with UI feedback"""
        try:
            if progress_callback:
                progress_callback(5, "Initializing quinté incremental training pipeline...")

            # Initialize pipeline
            pipeline = IncrementalTrainingPipeline(
                model_path=None,  # Use latest quinté model from config
                db_name=self.get_active_db(),
                verbose=True
            )

            if progress_callback:
                progress_callback(10, "Running quinté incremental training...")

            # Run quinté training pipeline
            results = pipeline.run_quinte_incremental_training(
                date_from=date_from,
                date_to=date_to,
                limit=limit,
                focus_on_failures=focus_on_failures,
                progress_callback=progress_callback
            )

            # Check if training was successful
            success = results.get("status") == "success"

            if not success:
                return {
                    "success": False,
                    "message": results.get("message", "Quinté incremental training failed"),
                    "error": results.get("error", "Unknown error"),
                    "training_results": results
                }

            # Format results for UI consumption
            formatted_results = {
                "success": True,
                "message": self._format_quinte_training_message(results),
                "training_results": {
                    "races_processed": results.get("races_processed", 0),
                    "baseline_metrics": results.get("baseline_metrics", {}),
                    "improved_metrics": results.get("improved_metrics", {}),
                    "failure_patterns": results.get("failure_patterns", {}),
                    "corrections_suggested": results.get("corrections_suggested", 0),
                    "corrections_applied": results.get("corrections_applied", []),
                    "model_saved": results.get("model_saved", ""),
                    "execution_time": results.get("execution_time", 0)
                }
            }

            return formatted_results

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "message": f"Quinté incremental training failed: {str(e)}",
                "traceback": traceback.format_exc(),
                "training_results": {}
            }

    def _format_quinte_training_message(self, results: Dict[str, Any]) -> str:
        """Format a user-friendly message for quinté training results"""
        races_count = results.get("races_processed", 0)

        message_parts = [f"Processed {races_count} quinté races"]

        # Baseline metrics
        baseline = results.get("baseline_metrics", {})
        improved = results.get("improved_metrics", {})

        if baseline and improved:
            # Quinté désordre improvement
            baseline_desordre = baseline.get("quinte_desordre_rate", 0) * 100
            improved_desordre = improved.get("quinte_desordre_rate", 0) * 100
            desordre_change = improved_desordre - baseline_desordre

            if desordre_change > 0:
                message_parts.append(f"Quinté désordre improved: {baseline_desordre:.1f}% → {improved_desordre:.1f}% (+{desordre_change:.1f}%)")
            else:
                message_parts.append(f"Quinté désordre: {improved_desordre:.1f}%")

            # Bonus 4 improvement
            baseline_bonus4 = baseline.get("bonus_4_rate", 0) * 100
            improved_bonus4 = improved.get("bonus_4_rate", 0) * 100
            bonus4_change = improved_bonus4 - baseline_bonus4

            if bonus4_change > 0:
                message_parts.append(f"Bonus 4 improved: {baseline_bonus4:.1f}% → {improved_bonus4:.1f}% (+{bonus4_change:.1f}%)")

        # Corrections applied
        corrections_count = results.get("corrections_suggested", 0)
        if corrections_count > 0:
            message_parts.append(f"Applied {corrections_count} correction strategies")

        # Model saved
        model_path = results.get("model_saved", "")
        if model_path:
            model_name = model_path.split('/')[-1] if '/' in model_path else "new model"
            message_parts.append(f"Saved: {model_name}")

        return ". ".join(message_parts)

    def get_races_with_results(self, date_from: str, date_to: str) -> List[Dict[str, Any]]:
        """Get races that have both predictions and results for the incremental training UI"""
        try:
            # Use the pipeline to get completed races
            pipeline = IncrementalTrainingPipeline(
                model_path=None,
                db_name=self.get_active_db(),
                verbose=False
            )

            races = pipeline.fetch_completed_races(date_from, date_to)
            return races

        except Exception as e:
            print(f"Error getting races with results: {e}")
            return []

    
    def get_prediction_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all prediction models"""
        try:
            # Get model paths directly from config without requiring PredictionOrchestrator
            model_paths = self.get_model_paths()

            # Get blend configuration from config (with fallbacks)
            blend_config = {}
            if self._config_data:
                blend_config = self._config_data.get('blend', {
                    'rf_weight': 0.5,
                    'tabnet_weight': 0.5,
                    'optimal_mae': 11.78
                })

            # Check if model files actually exist
            import os
            rf_ready = False
            tabnet_ready = False

            if 'rf' in model_paths:
                rf_model_file = f"{model_paths['rf']}/rf_model.joblib"
                rf_ready = os.path.exists(rf_model_file)

            if 'tabnet' in model_paths:
                tabnet_model_file = f"{model_paths['tabnet']}/tabnet_model.zip"
                tabnet_ready = os.path.exists(tabnet_model_file)

            # Format simplified model information
            prediction_info = {
                'ensemble_type': 'RF + TabNet Ensemble',
                'blend_weights': {
                    'rf_weight': blend_config.get('rf_weight', 0.5),
                    'tabnet_weight': blend_config.get('tabnet_weight', 0.5)
                },
                'optimal_mae': blend_config.get('optimal_mae', 11.78),
                'model_status': {
                    'rf_loaded': rf_ready,
                    'tabnet_loaded': tabnet_ready
                },
                'model_paths': model_paths,
                'system_status': 'ready' if (rf_ready and tabnet_ready) else 'partial' if (rf_ready or tabnet_ready) else 'no_models'
            }

            return {
                "success": True,
                "message": "Model information retrieved successfully",
                "prediction_info": prediction_info
            }

        except Exception as e:
            # Enhanced error handling with fallback information
            error_msg = str(e)

            return {
                "success": False,
                "error": error_msg,
                "message": f"Could not retrieve model information: {error_msg}",
                "prediction_info": {
                    "ensemble_type": "RF + TabNet Ensemble",
                    "blend_weights": {
                        "rf_weight": 0.5,
                        "tabnet_weight": 0.5
                    },
                    "model_status": {
                        "rf_loaded": False,
                        "tabnet_loaded": False
                    },
                    "model_paths": self.get_model_paths(),
                    "system_status": "error"
                }
            }
    
    def _check_alternative_models(self) -> Dict[str, bool]:
        """Check if alternative model files exist"""
        model_status = {
            'transformer_loaded': False,
            'ensemble_loaded': False
        }
        
        try:
            models_dir = Path("models")
            if models_dir.exists():
                # Check for transformer model files
                transformer_files = list(models_dir.glob("*transformer*.h5")) + list(models_dir.glob("*transformer*.keras"))
                model_status['transformer_loaded'] = len(transformer_files) > 0
                
                # Check for ensemble model files (pickle files)
                ensemble_files = list(models_dir.glob("*ensemble*.pkl")) + list(models_dir.glob("*ensemble*.joblib"))
                model_status['ensemble_loaded'] = len(ensemble_files) > 0
                
        except Exception as e:
            print(f"Warning: Could not check alternative model status: {e}")
            
        return model_status
    
    def _get_overall_system_status(self, legacy_model_info: Dict, alt_model_status: Dict) -> str:
        """Determine overall system status"""
        legacy_loaded = any(legacy_model_info.get(k, False) for k in ['has_rf', 'has_lstm', 'has_tabnet'])
        alt_loaded = any(alt_model_status.values())
        
        if legacy_loaded and alt_loaded:
            return 'full_operational'
        elif legacy_loaded:
            return 'legacy_operational'
        elif alt_loaded:
            return 'alternative_operational'
        else:
            return 'no_models_loaded'

    def get_ai_quinte_advice(self, lm_studio_url: str = None, verbose: bool = False) -> Dict[str, Any]:
        """Get AI betting advice specifically focused on quinte races with 3 refined recommendations"""
        try:
            # Initialize the evaluator to get comprehensive results
            evaluator = PredictEvaluator()
            
            # Get evaluation metrics
            metrics = evaluator.evaluate_all_races()
            
            # Get bet type wins
            bet_type_wins = evaluator.get_races_won_by_bet_type()
            
            # Get quinte analysis - this is the main focus
            quinte_analysis = evaluator.get_quinte_horse_betting_analysis()
            
            # Format evaluation results for AI advisor
            evaluation_results = self._format_results_for_advisor(metrics, bet_type_wins, quinte_analysis)
            
            # Initialize AI advisor (will use config if lm_studio_url is None)
            advisor = BettingAdvisor(lm_studio_url=lm_studio_url, verbose=verbose)
            
            # Get AI advice specifically for quinte
            ai_advice = advisor.analyze_quinte_betting_strategy(evaluation_results)
            
            return {
                "success": True,
                "message": "AI quinte betting advice generated successfully",
                "ai_advice": ai_advice,
                "evaluation_data": evaluation_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get AI quinte betting advice: {str(e)}"
            }

    def load_weight_analysis_data(self, date_from: str = None, date_to: str = None,
                                   race_filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load race predictions and metadata for weight analysis - simplified approach"""
        try:
            from utils.env_setup import AppConfig
            import sqlite3
            import pandas as pd
            import json

            config = AppConfig()
            db_path = config.get_active_db_path()
            conn = sqlite3.connect(db_path)

            # Get races with actual results (not pending)
            query = """
                SELECT comp, jour, typec, dist, partant, hippo, cheque, actual_results
                FROM daily_race
                WHERE actual_results IS NOT NULL
                  AND actual_results != 'pending'
                  AND actual_results != ''
            """

            params = []
            if date_from:
                query += " AND date(jour) >= ?"
                params.append(date_from)
            if date_to:
                query += " AND date(jour) <= ?"
                params.append(date_to)

            if race_filters:
                if race_filters.get('typec'):
                    query += " AND typec = ?"
                    params.append(race_filters['typec'])
                if race_filters.get('min_dist'):
                    query += " AND dist >= ?"
                    params.append(race_filters['min_dist'])
                if race_filters.get('max_dist'):
                    query += " AND dist <= ?"
                    params.append(race_filters['max_dist'])
                if race_filters.get('min_partant'):
                    query += " AND partant >= ?"
                    params.append(race_filters['min_partant'])
                if race_filters.get('max_partant'):
                    query += " AND partant <= ?"
                    params.append(race_filters['max_partant'])
                if race_filters.get('hippo'):
                    query += " AND hippo = ?"
                    params.append(race_filters['hippo'])

            races_df = pd.read_sql_query(query, conn, params=params)

            if races_df.empty:
                return {
                    "success": False,
                    "error": "No races found",
                    "message": "No races with results found in the specified date range"
                }

            # For each race: get predictions and create mapping
            race_data = []
            skipped_no_predictions = 0
            skipped_no_mapping = 0
            skipped_no_results = 0
            for _, race in races_df.iterrows():
                comp = race['comp']

                # Get the prediction_results JSON to get numero→idche mapping
                mapping_query = """
                    SELECT prediction_results
                    FROM daily_race
                    WHERE comp = ?
                """
                mapping_result = pd.read_sql_query(mapping_query, conn, params=[comp])

                if mapping_result.empty or not mapping_result.iloc[0]['prediction_results']:
                    skipped_no_mapping += 1
                    continue

                # Parse prediction_results to get numero→idche mapping
                try:
                    pred_json = json.loads(mapping_result.iloc[0]['prediction_results'])
                    numero_to_idche = {}
                    idche_to_numero = {}
                    for pred in pred_json.get('predictions', []):
                        numero = pred.get('numero')
                        idche = pred.get('idche')
                        if numero and idche:
                            numero_to_idche[numero] = idche
                            idche_to_numero[idche] = numero
                except (json.JSONDecodeError, KeyError):
                    skipped_no_mapping += 1
                    continue

                # Get raw RF/TabNet predictions for this race
                pred_query = """
                    SELECT horse_id, rf_prediction, tabnet_prediction
                    FROM race_predictions
                    WHERE race_id = ?
                      AND rf_prediction IS NOT NULL
                      AND tabnet_prediction IS NOT NULL
                """

                preds = pd.read_sql_query(pred_query, conn, params=[comp])

                if preds.empty:
                    skipped_no_predictions += 1
                    continue

                # Parse actual results (formato: "11-7-8-3..." these are numeros)
                actual_order = race['actual_results'].split('-')
                actual_positions = {}  # numero → finishing position
                for pos, numero_str in enumerate(actual_order, 1):
                    try:
                        actual_positions[int(numero_str)] = pos
                    except ValueError:
                        continue

                # Add numero to predictions using idche→numero mapping
                enriched_preds = []
                for pred in preds.to_dict('records'):
                    horse_id = pred['horse_id']
                    numero = idche_to_numero.get(horse_id)
                    if numero:
                        pred['numero'] = numero
                        enriched_preds.append(pred)

                if not enriched_preds:
                    continue

                # Store race with predictions and actual results
                race_data.append({
                    'comp': comp,
                    'jour': race['jour'],
                    'typec': race['typec'],
                    'dist': race['dist'],
                    'partant': race['partant'],
                    'hippo': race['hippo'],
                    'cheque': race['cheque'],
                    'predictions': enriched_preds,
                    'actual_positions': actual_positions  # numero → position
                })

            return {
                "success": True,
                "race_data": race_data,
                "message": f"Loaded {len(race_data)} races with predictions and results"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to load weight analysis data: {str(e)}"
            }

    def test_weight_combinations(self, data_df, weight_step: float = 0.1) -> Dict[str, Any]:
        """Test all RF/TabNet weight combinations and calculate metrics"""
        try:
            import numpy as np

            results = []

            # Generate weight combinations (RF + TabNet = 1.0)
            for rf_weight in np.arange(0.0, 1.01, weight_step):
                tabnet_weight = 1.0 - rf_weight

                # Calculate blended predictions
                data_df['blended_prediction'] = (
                    data_df['rf_prediction'] * rf_weight +
                    data_df['tabnet_prediction'] * tabnet_weight
                )

                # Group by race to calculate accuracy
                races_grouped = data_df.groupby('race_comp')

                winner_correct = 0
                podium_correct = 0
                total_races = 0

                mae_list = []
                rmse_list = []

                for race_comp, race_data in races_grouped:
                    total_races += 1

                    # Sort by blended prediction
                    race_data_sorted = race_data.sort_values('blended_prediction')
                    race_data_sorted['predicted_position'] = range(1, len(race_data_sorted) + 1)

                    # Check winner accuracy
                    predicted_winner = race_data_sorted.iloc[0]['numero']
                    actual_winner_data = race_data[race_data['actual_result'] == 1]
                    if len(actual_winner_data) > 0:
                        actual_winner = actual_winner_data.iloc[0]['numero']
                        if predicted_winner == actual_winner:
                            winner_correct += 1

                    # Check podium accuracy (top 3)
                    predicted_podium = set(race_data_sorted.head(3)['numero'].values)
                    actual_podium_data = race_data[race_data['actual_result'] <= 3]
                    if len(actual_podium_data) >= 3:
                        actual_podium = set(actual_podium_data['numero'].values)
                        if len(predicted_podium & actual_podium) == 3:
                            podium_correct += 1

                    # Calculate MAE and RMSE
                    for _, row in race_data_sorted.iterrows():
                        predicted_pos = row['predicted_position']
                        actual_pos = row['actual_result']
                        error = abs(predicted_pos - actual_pos)
                        mae_list.append(error)
                        rmse_list.append(error ** 2)

                # Calculate aggregate metrics
                winner_accuracy = winner_correct / total_races if total_races > 0 else 0
                podium_accuracy = podium_correct / total_races if total_races > 0 else 0
                mae = np.mean(mae_list) if mae_list else 0
                rmse = np.sqrt(np.mean(rmse_list)) if rmse_list else 0

                results.append({
                    'rf_weight': round(rf_weight, 2),
                    'tabnet_weight': round(tabnet_weight, 2),
                    'winner_accuracy': winner_accuracy,
                    'podium_accuracy': podium_accuracy,
                    'mae': mae,
                    'rmse': rmse,
                    'total_races': total_races
                })

            return {
                "success": True,
                "results": results,
                "message": f"Tested {len(results)} weight combinations"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to test weight combinations: {str(e)}"
            }

    def analyze_by_race_characteristics(self, data_df, optimal_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze performance breakdown by race characteristics"""
        try:
            import numpy as np

            if optimal_weights is None:
                optimal_weights = {'rf_weight': 0.8, 'tabnet_weight': 0.2}

            # Calculate blended predictions with optimal weights
            data_df['blended_prediction'] = (
                data_df['rf_prediction'] * optimal_weights['rf_weight'] +
                data_df['tabnet_prediction'] * optimal_weights['tabnet_weight']
            )

            analysis_results = {}

            # Analysis by race type (typec)
            analysis_results['by_typec'] = self._analyze_by_characteristic(
                data_df, 'typec', optimal_weights
            )

            # Analysis by distance buckets
            data_df['dist_bucket'] = pd.cut(
                data_df['dist'],
                bins=[0, 1500, 2000, 2500, 3000, 10000],
                labels=['<1500m', '1500-2000m', '2000-2500m', '2500-3000m', '>3000m']
            )
            analysis_results['by_distance'] = self._analyze_by_characteristic(
                data_df, 'dist_bucket', optimal_weights
            )

            # Analysis by field size buckets
            data_df['field_bucket'] = pd.cut(
                data_df['partant'],
                bins=[0, 8, 12, 16, 30],
                labels=['Small (≤8)', 'Medium (9-12)', 'Large (13-16)', 'Very Large (>16)']
            )
            analysis_results['by_field_size'] = self._analyze_by_characteristic(
                data_df, 'field_bucket', optimal_weights
            )

            # Analysis by purse level buckets
            data_df['purse_bucket'] = pd.cut(
                data_df['montant'],
                bins=[0, 20000, 50000, 100000, 1000000],
                labels=['Low (<20k)', 'Medium (20-50k)', 'High (50-100k)', 'Premium (>100k)']
            )
            analysis_results['by_purse'] = self._analyze_by_characteristic(
                data_df, 'purse_bucket', optimal_weights
            )

            # Top 5 hippodromes by race count
            top_hippos = data_df['hippo'].value_counts().head(5).index.tolist()
            data_df_top_hippos = data_df[data_df['hippo'].isin(top_hippos)]
            analysis_results['by_hippo'] = self._analyze_by_characteristic(
                data_df_top_hippos, 'hippo', optimal_weights
            )

            return {
                "success": True,
                "analysis": analysis_results,
                "message": "Race characteristic analysis completed"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to analyze by race characteristics: {str(e)}"
            }

    def _analyze_by_characteristic(self, data_df, characteristic: str, optimal_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Helper function to analyze performance by a specific characteristic"""
        import numpy as np

        results = []

        for char_value, char_data in data_df.groupby(characteristic):
            if pd.isna(char_value):
                continue

            races_grouped = char_data.groupby('race_comp')

            winner_correct = 0
            podium_correct = 0
            total_races = 0
            mae_list = []

            for race_comp, race_data in races_grouped:
                total_races += 1

                # Sort by blended prediction
                race_data_sorted = race_data.sort_values('blended_prediction')
                race_data_sorted['predicted_position'] = range(1, len(race_data_sorted) + 1)

                # Check winner accuracy
                predicted_winner = race_data_sorted.iloc[0]['numero']
                actual_winner_data = race_data[race_data['actual_result'] == 1]
                if len(actual_winner_data) > 0:
                    actual_winner = actual_winner_data.iloc[0]['numero']
                    if predicted_winner == actual_winner:
                        winner_correct += 1

                # Check podium accuracy
                predicted_podium = set(race_data_sorted.head(3)['numero'].values)
                actual_podium_data = race_data[race_data['actual_result'] <= 3]
                if len(actual_podium_data) >= 3:
                    actual_podium = set(actual_podium_data['numero'].values)
                    if len(predicted_podium & actual_podium) == 3:
                        podium_correct += 1

                # Calculate MAE
                for _, row in race_data_sorted.iterrows():
                    predicted_pos = row['predicted_position']
                    actual_pos = row['actual_result']
                    mae_list.append(abs(predicted_pos - actual_pos))

            winner_accuracy = winner_correct / total_races if total_races > 0 else 0
            podium_accuracy = podium_correct / total_races if total_races > 0 else 0
            mae = np.mean(mae_list) if mae_list else 0

            results.append({
                'characteristic_value': str(char_value),
                'winner_accuracy': winner_accuracy,
                'podium_accuracy': podium_accuracy,
                'mae': mae,
                'total_races': total_races
            })

        return results

    def _test_weights_on_races(self, race_data: List[Dict], weight_step: float = 0.1) -> List[Dict]:
        """Test all weight combinations on a set of races"""
        import numpy as np

        results = []

        for rf_weight in np.arange(0.0, 1.01, weight_step):
            tabnet_weight = 1.0 - rf_weight

            winner_correct = 0
            podium_correct = 0
            total_races = len(race_data)
            mae_list = []

            for race in race_data:
                predictions = race['predictions']
                actual_positions = race['actual_positions']

                # Calculate blended prediction for each horse
                horse_predictions = []
                for pred in predictions:
                    numero = pred['numero']  # Now using numero instead of horse_id
                    blended = (pred['rf_prediction'] * rf_weight +
                              pred['tabnet_prediction'] * tabnet_weight)
                    horse_predictions.append({
                        'numero': numero,
                        'blended_prediction': blended
                    })

                # Sort by blended prediction (lower = better position)
                horse_predictions.sort(key=lambda x: x['blended_prediction'])

                # Assign predicted positions
                for rank, hp in enumerate(horse_predictions, 1):
                    hp['predicted_position'] = rank

                # Check winner accuracy
                if horse_predictions:
                    predicted_winner = horse_predictions[0]['numero']
                    # Find actual winner (position 1 in actual_positions which maps numero→position)
                    actual_winner = next((num for num, pos in actual_positions.items() if pos == 1), None)
                    if predicted_winner == actual_winner:
                        winner_correct += 1

                # Check podium accuracy (top 3) - count if at least 2 of 3 are correct
                # Balanced metric: how often do we get at least 2 podium finishers in our top 3
                predicted_podium = {hp['numero'] for hp in horse_predictions[:3]}
                actual_podium = {num for num, pos in actual_positions.items() if pos <= 3}
                overlap = len(predicted_podium & actual_podium)
                if overlap >= 2:  # At least 2 correct podium horses in top 3
                    podium_correct += 1

                # Calculate MAE
                for hp in horse_predictions:
                    actual_pos = actual_positions.get(hp['numero'])
                    if actual_pos:
                        mae_list.append(abs(hp['predicted_position'] - actual_pos))

            # Calculate aggregate metrics
            results.append({
                'rf_weight': round(rf_weight, 2),
                'tabnet_weight': round(tabnet_weight, 2),
                'winner_accuracy': winner_correct / total_races if total_races > 0 else 0,
                'podium_accuracy': podium_correct / total_races if total_races > 0 else 0,
                'mae': np.mean(mae_list) if mae_list else 0,
                'rmse': np.sqrt(np.mean([e**2 for e in mae_list])) if mae_list else 0,
                'total_races': total_races
            })

        return results

    def detect_weight_patterns(self, race_data: List[Dict], weight_step: float = 0.1) -> Dict[str, Any]:
        """Automated pattern detection: test all weights and find optimal patterns by race features"""
        try:
            import numpy as np

            # Step 1: Test all weight combinations on all races
            overall_results = self._test_weights_on_races(race_data, weight_step)

            # Step 2: Find overall best weights
            best_overall = max(overall_results, key=lambda x: x['winner_accuracy'])

            # Step 3: Test different weights for each race characteristic
            patterns = {
                'overall_best': {
                    'rf_weight': best_overall['rf_weight'],
                    'tabnet_weight': best_overall['tabnet_weight'],
                    'winner_accuracy': best_overall['winner_accuracy'],
                    'podium_accuracy': best_overall['podium_accuracy'],
                    'mae': best_overall['mae'],
                    'total_races': best_overall['total_races']
                },
                'by_race_type': [],
                'by_distance_range': [],
                'by_field_size': [],
                'by_purse_level': [],
                'summary': []
            }

            # Pattern detection for race types
            typec_groups = {}
            for race in race_data:
                typec = race.get('typec')
                if typec:
                    if typec not in typec_groups:
                        typec_groups[typec] = []
                    typec_groups[typec].append(race)

            for typec, races in typec_groups.items():
                if len(races) < 10:  # Skip if too few races
                    continue

                typec_results = self._test_weights_on_races(races, weight_step)
                best_for_typec = max(typec_results, key=lambda x: x['winner_accuracy'])

                # Only include if weights differ from overall by 0.2+ OR accuracy improvement > 5%
                weight_diff = abs(best_for_typec['rf_weight'] - best_overall['rf_weight'])
                accuracy_improvement = best_for_typec['winner_accuracy'] - best_overall['winner_accuracy']

                if weight_diff >= 0.2 or accuracy_improvement > 0.05:
                    patterns['by_race_type'].append({
                        'typec': typec,
                        'optimal_rf_weight': best_for_typec['rf_weight'],
                        'optimal_tabnet_weight': best_for_typec['tabnet_weight'],
                        'winner_accuracy': best_for_typec['winner_accuracy'],
                        'improvement_vs_overall': accuracy_improvement,
                        'total_races': best_for_typec['total_races'],
                        'recommendation': 'Use custom weights' if weight_diff >= 0.2 else 'Overall weights work well'
                    })

            # Pattern detection for distance ranges
            def get_distance_bucket(dist):
                if dist < 1500:
                    return '<1500m'
                elif dist < 2000:
                    return '1500-2000m'
                elif dist < 2500:
                    return '2000-2500m'
                elif dist < 3000:
                    return '2500-3000m'
                else:
                    return '>3000m'

            dist_groups = {}
            for race in race_data:
                dist = race.get('dist', 0)
                bucket = get_distance_bucket(dist)
                if bucket not in dist_groups:
                    dist_groups[bucket] = []
                dist_groups[bucket].append(race)

            for dist_range, races in dist_groups.items():
                if len(races) < 10:
                    continue

                dist_results = self._test_weights_on_races(races, weight_step)
                best_for_dist = max(dist_results, key=lambda x: x['winner_accuracy'])

                weight_diff = abs(best_for_dist['rf_weight'] - best_overall['rf_weight'])
                accuracy_improvement = best_for_dist['winner_accuracy'] - best_overall['winner_accuracy']

                if weight_diff >= 0.2 or accuracy_improvement > 0.05:
                    patterns['by_distance_range'].append({
                        'distance_range': str(dist_range),
                        'optimal_rf_weight': best_for_dist['rf_weight'],
                        'optimal_tabnet_weight': best_for_dist['tabnet_weight'],
                        'winner_accuracy': best_for_dist['winner_accuracy'],
                        'improvement_vs_overall': accuracy_improvement,
                        'total_races': best_for_dist['total_races'],
                        'recommendation': 'Use custom weights' if weight_diff >= 0.2 else 'Overall weights work well'
                    })

            # Pattern detection for field size
            def get_field_bucket(partant):
                if partant <= 8:
                    return 'Small (≤8)'
                elif partant <= 12:
                    return 'Medium (9-12)'
                elif partant <= 16:
                    return 'Large (13-16)'
                else:
                    return 'Very Large (>16)'

            field_groups = {}
            for race in race_data:
                partant = race.get('partant', 0)
                bucket = get_field_bucket(partant)
                if bucket not in field_groups:
                    field_groups[bucket] = []
                field_groups[bucket].append(race)

            for field_size, races in field_groups.items():
                if len(races) < 10:
                    continue

                field_results = self._test_weights_on_races(races, weight_step)
                best_for_field = max(field_results, key=lambda x: x['winner_accuracy'])

                weight_diff = abs(best_for_field['rf_weight'] - best_overall['rf_weight'])
                accuracy_improvement = best_for_field['winner_accuracy'] - best_overall['winner_accuracy']

                if weight_diff >= 0.2 or accuracy_improvement > 0.05:
                    patterns['by_field_size'].append({
                        'field_size': str(field_size),
                        'optimal_rf_weight': best_for_field['rf_weight'],
                        'optimal_tabnet_weight': best_for_field['tabnet_weight'],
                        'winner_accuracy': best_for_field['winner_accuracy'],
                        'improvement_vs_overall': accuracy_improvement,
                        'total_races': best_for_field['total_races'],
                        'recommendation': 'Use custom weights' if weight_diff >= 0.2 else 'Overall weights work well'
                    })

            # Generate summary insights
            total_patterns = len(patterns['by_race_type']) + len(patterns['by_distance_range']) + len(patterns['by_field_size'])

            if total_patterns == 0:
                patterns['summary'].append({
                    'type': 'no_patterns',
                    'message': f"Overall weights ({best_overall['rf_weight']:.1f} RF / {best_overall['tabnet_weight']:.1f} TabNet) work well across all race types"
                })
            else:
                patterns['summary'].append({
                    'type': 'patterns_found',
                    'message': f"Found {total_patterns} significant patterns requiring custom weights"
                })

            if len(patterns['by_race_type']) > 0:
                patterns['summary'].append({
                    'type': 'race_type_patterns',
                    'message': f"Race type matters: {len(patterns['by_race_type'])} types benefit from custom weights"
                })

            if len(patterns['by_distance_range']) > 0:
                patterns['summary'].append({
                    'type': 'distance_patterns',
                    'message': f"Distance matters: {len(patterns['by_distance_range'])} ranges benefit from custom weights"
                })

            if len(patterns['by_field_size']) > 0:
                patterns['summary'].append({
                    'type': 'field_size_patterns',
                    'message': f"Field size matters: {len(patterns['by_field_size'])} sizes benefit from custom weights"
                })

            return {
                "success": True,
                "patterns": patterns,
                "all_weight_results": overall_results,
                "message": f"Pattern detection complete: found {total_patterns} significant patterns"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to detect patterns: {str(e)}"
            }