import yaml
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
import threading
import queue
import sys
from pathlib import Path
import numpy as np

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
from core.storage.prediction_storage import PredictionStorage

class PipelineHelper:
    """Helper class for config management and status calculations"""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self._config_data = self.load_config()

        # Training management
        self.training_queue = queue.Queue()
        self.training_thread = None
        self.is_training = False
        
        # Initialize prediction storage
        try:
            app_config = AppConfig()
            self.prediction_storage = PredictionStorage(app_config)
        except Exception as e:
            print(f"Warning: Could not initialize prediction storage: {e}")
            self.prediction_storage = None

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
        print(f"Config data: {self._config_data}")  # Debug line
        if self._config_data and 'base' in self._config_data:
            print(f"Base section: {self._config_data['base']}")  # Debug line
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
        latest_model = (models.get('latest_model'))

        if latest_model:
            # Extract date from model name like "5years_hybrid_v20250606_182331"
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', latest_model)
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

                    version = latest_model.split('_')[-2] + '_' + latest_model.split('_')[-1]
                    return last_training, version
                except:
                    pass

        return 'Unknown', 'Unknown'

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
            'lstm': self._config_data.get('lstm', {}),
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
        self.training_thread = threading.Thread(
            target=self._training_worker,
            daemon=True
        )
        self.training_thread.start()
        return True

    def _training_worker(self):
        """Background training worker"""
        try:
            def progress_callback(percent, message):
                self.training_queue.put({
                    'type': 'progress',
                    'percent': percent,
                    'message': message
                })

            # Run actual training
            train_race_model.main(progress_callback=progress_callback)

            # Send completion
            self.training_queue.put({
                'type': 'complete',
                'success': True,
                'message': 'Training completed successfully'
            })
        except Exception as e:
            self.training_queue.put({
                'type': 'complete',
                'success': False,
                'error': str(e),
                'message': f'Training failed: {str(e)}'
            })
        finally:
            self.is_training = False

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
                            force_reprediction: bool = False) -> Dict[str, Any]:
        """Execute predictions for specified races or all unpredicted races"""
        try:
            if progress_callback:
                progress_callback(5, "Initializing prediction orchestrator...")

            # Initialize prediction orchestrator
            predictor = PredictionOrchestrator()
            
            # Get model information for debugging
            model_info = predictor.get_model_info()
            models_loaded = model_info.get('models_loaded', {})
            model_status = model_info.get('model_status', {})
            model_errors = model_info.get('model_errors', {})
            effective_weights = model_info.get('effective_weights', {})
            
            # Enhanced debug output showing which models are loaded and any errors
            available_models = model_info.get('available_models', [])
            debug_message = f"Models loaded: RF={models_loaded.get('rf', False)}, LSTM={models_loaded.get('lstm', False)}, TabNet={models_loaded.get('tabnet', False)}"
            print(f"[DEBUG] {debug_message}")
            
            # Log model errors and special statuses
            if model_errors:
                for model, error in model_errors.items():
                    # Handle TabNet fallback status specially
                    if model == 'tabnet' and model_status.get('tabnet') == 'loaded_fallback':
                        print(f"[INFO] TabNet using fallback model: {error}")
                    else:
                        print(f"[WARNING] {model.upper()} Model Issue: {error}")
            
            # Log effective weights 
            if effective_weights:
                weights_str = ", ".join([f"{model.upper()}={weight:.3f}" for model, weight in effective_weights.items()])
                print(f"[INFO] Effective blend weights: {weights_str}")
            
            # Check for TabNet fallback and display clear message
            if model_status.get('tabnet') == 'loaded_fallback':
                print(f"[INFO] ✅ All 3 models loaded successfully (TabNet using fallback from previous model)")
            elif len(available_models) == 3:
                print(f"[INFO] ✅ All 3 models loaded successfully")
            else:
                # Warning if not all 3 models are available
                missing_models = [model for model in ['rf', 'lstm', 'tabnet'] if model not in available_models]
                print(f"[WARNING] ❌ Missing models: {', '.join(missing_models).upper()}. Using {len(available_models)}/3 models for predictions.")
            
            if progress_callback:
                progress_callback(10, f"Getting races to predict... {debug_message}")

            # Get races to predict
            if races_to_predict:
                # Predict specific races
                total_races = len(races_to_predict)
                predicted_count = 0

                for i, comp in enumerate(races_to_predict):
                    if progress_callback:
                        progress = 10 + (i / total_races) * 80
                        progress_callback(int(progress), f"Predicting race {comp}...")

                    try:
                        result = predictor.predict_race(comp)
                        if result.get('status') == 'success':
                            predicted_count += 1
                    except Exception as e:
                        print(f"Error predicting race {comp}: {str(e)}")
                        continue
            else:
                # Predict races based on force_reprediction flag
                if force_reprediction:
                    # Get all races with processed data (ignore existing predictions)
                    races_to_process = self.get_races_for_reprediction()
                    message_type = "all processed races (force reprediction)"
                else:
                    # Get only unpredicted races
                    races_to_process = self.get_races_needing_prediction()
                    message_type = "unpredicted races"

                total_races = len(races_to_process)
                predicted_count = 0

                if total_races == 0:
                    if progress_callback:
                        progress_callback(100, f"No {message_type} found")
                    return {
                        "success": True,
                        "message": f"No {message_type} found",
                        "predicted_count": 0,
                        "total_races": 0
                    }

                for i, race in enumerate(races_to_process):
                    comp = race['comp']
                    if progress_callback:
                        progress = 10 + (i / total_races) * 80
                        progress_callback(int(progress), f"Predicting race {comp}...")

                    try:
                        result = predictor.predict_race(comp)
                        if result.get('status') == 'success':
                            predicted_count += 1
                    except Exception as e:
                        print(f"Error predicting race {comp}: {str(e)}")
                        continue

                races_to_predict = [race['comp'] for race in races_to_process]

            if progress_callback:
                progress_callback(100, f"Predictions completed: {predicted_count}/{len(races_to_predict)} successful")

            # Add model debug info to the final result
            model_summary = f"Using {len(available_models)}/3 models: {', '.join(available_models).upper()}"
            final_message = f"Predictions completed: {predicted_count}/{len(races_to_predict)} successful. {model_summary}"
            
            return {
                "success": True,
                "message": final_message,
                "predicted_count": predicted_count,
                "total_races": len(races_to_predict),
                "model_info": model_info  # Include full model info in response
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Prediction failed: {str(e)}"
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
                    "lstm_training": results.get("lstm_training", {}),
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

        # LSTM training results
        lstm_results = results.get("lstm_training", {})
        if lstm_results.get("status") == "success":
            lstm_improvement = lstm_results.get("improvement", {})
            mae_improvement = lstm_improvement.get("mae_improvement_pct", 0)
            message_parts.append(f"LSTM model improved by {mae_improvement:.1f}%")
        elif lstm_results.get("status") == "skipped":
            message_parts.append("LSTM training skipped")

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

    def get_ai_betting_advice(self, lm_studio_url: str = None, verbose: bool = False) -> Dict[str, Any]:
        """Get AI betting advice based on latest evaluation results"""
        try:
            # Initialize the evaluator to get comprehensive results
            evaluator = PredictEvaluator()
            
            # Get evaluation metrics
            metrics = evaluator.evaluate_all_races()
            
            # Get bet type wins
            bet_type_wins = evaluator.get_races_won_by_bet_type()
            
            # Get quinte analysis
            quinte_analysis = evaluator.get_quinte_horse_betting_analysis()
            
            # Format evaluation results for AI advisor
            evaluation_results = self._format_results_for_advisor(metrics, bet_type_wins, quinte_analysis)
            
            # Initialize AI advisor (will use config if lm_studio_url is None)
            advisor = BettingAdvisor(lm_studio_url=lm_studio_url, verbose=verbose)
            
            # Get AI advice
            ai_advice = advisor.analyze_daily_results(evaluation_results)
            
            return {
                "success": True,
                "message": "AI betting advice generated successfully",
                "ai_advice": ai_advice,
                "evaluation_data": evaluation_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get AI betting advice: {str(e)}"
            }

    def get_ai_race_advice(self, race_comp: str, lm_studio_url: str = None, verbose: bool = False) -> Dict[str, Any]:
        """Get AI advice for a specific race"""
        try:
            # Get race data and predictions
            race_fetcher = RaceFetcher()
            race_data = race_fetcher.get_race_by_comp(race_comp)
            
            if not race_data:
                return {
                    "success": False,
                    "message": f"Race {race_comp} not found"
                }
            
            # Get prediction results from race data
            prediction_results = race_data.get('prediction_results')
            
            if not prediction_results:
                return {
                    "success": False,
                    "message": f"No predictions found for race {race_comp}"
                }
            
            # Parse prediction results if they're stored as JSON string
            if isinstance(prediction_results, str):
                try:
                    prediction_results = json.loads(prediction_results)
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "message": f"Invalid prediction results format for race {race_comp}"
                    }
            
            # Get previous results for context
            evaluator = PredictEvaluator()
            previous_results = evaluator.evaluate_all_races()
            previous_results_dict = self._format_results_for_advisor(previous_results, {}, {})
            
            # Initialize AI advisor (will use config if lm_studio_url is None)
            advisor = BettingAdvisor(lm_studio_url=lm_studio_url, verbose=verbose)
            
            # Get AI race advice
            ai_advice = advisor.analyze_race_prediction(race_data, prediction_results, previous_results_dict)
            
            return {
                "success": True,
                "message": "AI race advice generated successfully",
                "ai_advice": ai_advice,
                "race_data": race_data,
                "predictions": prediction_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get AI race advice: {str(e)}"
            }

    def _format_results_for_advisor(self, metrics, bet_type_wins, quinte_analysis) -> Dict[str, Any]:
        """Format evaluation results for AI advisor consumption"""
        # Convert metrics object to dictionary format expected by advisor
        formatted_results = {
            'summary_metrics': {
                'total_races': metrics.total_races,
                'winner_accuracy': metrics.overall_winner_accuracy,
                'podium_accuracy': metrics.overall_podium_accuracy,
                'mean_rank_error': getattr(metrics, 'mean_rank_error', 0)
            },
            'pmu_summary': {},
            'race_details': []
        }
        
        # Format bet type wins into PMU summary format
        if hasattr(metrics, 'bet_win_rates'):
            for bet_type, stats in metrics.bet_win_rates.items():
                formatted_results['pmu_summary'][f'{bet_type}_rate'] = stats.get('rate', 0)
        
        # Add quinte analysis
        if quinte_analysis:
            formatted_results['quinte_analysis'] = quinte_analysis
            
        # Add race details if available
        if bet_type_wins:
            formatted_results['race_details'] = bet_type_wins
            
        return formatted_results

    def get_prediction_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the three-model prediction system"""
        try:
            # Initialize prediction orchestrator to get model info
            predictor = PredictionOrchestrator()
            model_info = predictor.get_model_info()
            
            # Get blend configuration from config
            blend_config = self._config_data.get('blend', {}) if self._config_data else {}
            
            # Format comprehensive model information
            prediction_info = {
                'ensemble_type': 'Three-Model Ensemble (RF + LSTM + TabNet)',
                'blend_weights': {
                    'rf_weight': blend_config.get('rf_weight', 0.8),
                    'lstm_weight': blend_config.get('lstm_weight', 0.1),
                    'tabnet_weight': blend_config.get('tabnet_weight', 0.1)
                },
                'optimal_mae': blend_config.get('optimal_mae', 11.78),
                'model_status': model_info.get('model_status', {}),
                'model_errors': model_info.get('model_errors', {}),
                'effective_weights': model_info.get('effective_weights', {}),
                'model_path': model_info.get('model_path', 'Unknown'),
                'tabnet_info': {
                    'available': model_info.get('tabnet_available', False),
                    'installed': model_info.get('tabnet_available', False),
                    'status': model_info.get('model_status', {}).get('tabnet', 'unknown'),
                    'fallback_used': model_info.get('model_status', {}).get('tabnet') == 'loaded_fallback',
                    'fallback_path': model_info.get('model_errors', {}).get('tabnet', '') if model_info.get('model_status', {}).get('tabnet') == 'loaded_fallback' else None
                },
                'available_models': model_info.get('available_models', []),
                'available_models_count': model_info.get('available_models_count', 0),
                'system_status': self._determine_system_status(model_info)
            }
            
            return {
                "success": True,
                "message": "Model information retrieved successfully",
                "prediction_info": prediction_info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get model information: {str(e)}"
            }
    
    def _determine_system_status(self, model_info: Dict[str, Any]) -> str:
        """Determine system status based on model availability and issues."""
        available_models_count = model_info.get('available_models_count', 0)
        model_status = model_info.get('model_status', {})
        
        if available_models_count == 0:
            return 'no_models_loaded'
        elif available_models_count == 3:
            # Check if TabNet is using fallback
            if model_status.get('tabnet') == 'loaded_fallback':
                return 'operational_with_fallback'
            else:
                return 'fully_operational'
        elif available_models_count >= 2:
            return 'operational_degraded'
        else:
            return 'minimal_operation'

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

    # Prediction Storage Analytics Methods
    def get_prediction_storage_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get prediction storage performance analytics"""
        if not self.prediction_storage:
            return {"error": "Prediction storage not initialized"}
        
        try:
            performance = self.prediction_storage.get_recent_performance(days)
            return {
                "success": True,
                "data": performance
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_prediction_storage_bias_analysis(self, days: int = 60) -> Dict[str, Any]:
        """Get prediction storage bias analysis"""
        if not self.prediction_storage:
            return {"error": "Prediction storage not initialized"}
        
        try:
            bias_analysis = self.prediction_storage.analyze_model_bias(days)
            return {
                "success": True,
                "data": bias_analysis
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_prediction_storage_blend_suggestions(self, days: int = 30) -> Dict[str, Any]:
        """Get blend weight optimization suggestions"""
        if not self.prediction_storage:
            return {"error": "Prediction storage not initialized"}
        
        try:
            suggestions = self.prediction_storage.suggest_blend_weights(days)
            return {
                "success": True,
                "data": suggestions
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_prediction_storage_confidence_calibration(self, days: int = 30) -> Dict[str, Any]:
        """Get model confidence calibration analysis"""
        if not self.prediction_storage:
            return {"error": "Prediction storage not initialized"}
        
        try:
            calibration = self.prediction_storage.get_model_confidence_calibration(days)
            return {
                "success": True,
                "data": calibration
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def export_prediction_storage_data(self, days: int = 30) -> Dict[str, Any]:
        """Export prediction storage data for analysis"""
        if not self.prediction_storage:
            return {"error": "Prediction storage not initialized"}
        
        try:
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            export_path = self.prediction_storage.export_for_analysis(start_date, end_date)
            return {
                "success": True,
                "export_path": export_path,
                "message": f"Data exported to {export_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_prediction_storage_training_feedback(self, days: int = 7) -> Dict[str, Any]:
        """Get training feedback data from prediction storage"""
        if not self.prediction_storage:
            return {"error": "Prediction storage not initialized"}
        
        try:
            feedback_data = self.prediction_storage.get_training_feedback(days)
            return {
                "success": True,
                "data": feedback_data
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_prediction_storage_summary(self) -> Dict[str, Any]:
        """Get summary statistics from prediction storage"""
        if not self.prediction_storage:
            return {"error": "Prediction storage not initialized"}
        
        try:
            # Get recent performance (7 days)
            recent_performance = self.prediction_storage.get_recent_performance(7)
            
            # Get monthly performance for comparison
            monthly_performance = self.prediction_storage.get_recent_performance(30)
            
            # Get training feedback count
            feedback_data = self.prediction_storage.get_training_feedback(7)
            
            summary = {
                "recent_predictions": recent_performance.get("total_predictions", 0) if "error" not in recent_performance else 0,
                "monthly_predictions": monthly_performance.get("total_predictions", 0) if "error" not in monthly_performance else 0,
                "recent_ensemble_mae": recent_performance.get("ensemble_prediction_mae", "N/A") if "error" not in recent_performance else "N/A",
                "training_feedback_available": len(feedback_data) if isinstance(feedback_data, list) else 0,
                "storage_initialized": True,
                "database_path": self.prediction_storage.db_path
            }
            
            return {
                "success": True,
                "data": summary
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # Continuous Learning Methods for Phase 1 Foundation
    def store_daily_predictions(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """Manual trigger to store predictions from latest daily run into prediction storage"""
        try:
            if not self.prediction_storage:
                return {
                    "success": False,
                    "error": "Prediction storage not initialized"
                }
            
            # Get daily races with predictions
            daily_races = self.get_daily_races()
            
            if not daily_races:
                return {
                    "success": False,
                    "message": "No daily races found"
                }
            
            # Filter races with predictions
            races_with_predictions = [race for race in daily_races if race.get('has_predictions', 0) == 1]
            
            if not races_with_predictions:
                return {
                    "success": False,
                    "message": "No races with predictions found"
                }
            
            from core.storage.prediction_storage import PredictionRecord
            from core.connectors.api_daily_sync import RaceFetcher
            import json
            
            # Initialize race fetcher for getting detailed race data
            race_fetcher = RaceFetcher()
            stored_count = 0
            error_count = 0
            
            for race in races_with_predictions:
                try:
                    race_comp = race['comp']
                    
                    # Get detailed race data with predictions
                    detailed_race = race_fetcher.get_race_by_comp(race_comp)
                    if not detailed_race or not detailed_race.get('prediction_results'):
                        continue
                    
                    # Parse prediction results
                    prediction_results = detailed_race['prediction_results']
                    if isinstance(prediction_results, str):
                        try:
                            prediction_results = json.loads(prediction_results)
                        except json.JSONDecodeError:
                            error_count += 1
                            continue
                    
                    # Extract predictions for each horse
                    if 'predictions' in prediction_results:
                        for pred in prediction_results['predictions']:
                            horse_id = str(pred.get('numero', ''))
                            if not horse_id:
                                continue
                            
                            # Create prediction record
                            record = PredictionRecord(
                                race_id=race_comp,
                                timestamp=datetime.now(),
                                horse_id=horse_id,
                                rf_prediction=pred.get('rf_prediction'),
                                lstm_prediction=pred.get('lstm_prediction'),
                                tabnet_prediction=pred.get('tabnet_prediction'),
                                ensemble_prediction=pred.get('predicted_rank', pred.get('predicted_position', 0)),
                                ensemble_confidence_score=pred.get('ensemble_confidence_score', pred.get('confidence', pred.get('predicted_prob'))),
                                actual_position=pred.get('actual_position'),  # Will be None for future races
                                distance=race.get('distance'),
                                track_condition=race.get('etat_terrain'),
                                weather=race.get('meteo'),
                                field_size=race.get('partant'),
                                race_type=race.get('typec'),
                                model_versions={'stored_at': datetime.now().isoformat()},
                                prediction_metadata={
                                    'race_date': race.get('jour'),
                                    'track': race.get('hippo'),
                                    'race_number': race.get('prix'),
                                    'quinte': bool(race.get('quinte', 0))
                                }
                            )
                            
                            # Store the record
                            self.prediction_storage.store_prediction(record)
                            stored_count += 1
                            
                except Exception as e:
                    print(f"Error processing race {race.get('comp', 'unknown')}: {str(e)}")
                    error_count += 1
                    continue
            
            return {
                "success": True,
                "message": f"Stored {stored_count} predictions from {len(races_with_predictions)} races",
                "predictions_stored": stored_count,
                "races_processed": len(races_with_predictions),
                "errors": error_count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to store daily predictions: {str(e)}"
            }
    
    def show_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Display recent model performance metrics from prediction storage"""
        try:
            if not self.prediction_storage:
                return {
                    "success": False,
                    "error": "Prediction storage not initialized"
                }
            
            # Get performance metrics
            performance = self.prediction_storage.get_recent_performance(days)
            
            if "error" in performance:
                return {
                    "success": False,
                    "error": performance["error"]
                }
            
            # Get bias analysis
            bias_analysis = self.prediction_storage.analyze_model_bias(days)
            
            # Get blend suggestions
            blend_suggestions = self.prediction_storage.suggest_blend_weights(days)
            
            return {
                "success": True,
                "performance_metrics": performance,
                "bias_analysis": bias_analysis,
                "blend_suggestions": blend_suggestions,
                "message": f"Performance analysis for last {days} days completed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get performance metrics: {str(e)}"
            }
    
    def run_weekly_delta_training(self, days_back: int = 7, progress_callback=None) -> Dict[str, Any]:
        """Manual trigger for regression enhancement on recent prediction feedback data"""
        try:
            if progress_callback:
                progress_callback(10, "Initializing weekly delta training...")
            
            # Get training feedback data from prediction storage
            if not self.prediction_storage:
                return {
                    "success": False,
                    "error": "Prediction storage not initialized"
                }
            
            feedback_data = self.prediction_storage.get_training_feedback(days_back)
            
            if not feedback_data:
                return {
                    "success": False,
                    "message": f"No training feedback data found for last {days_back} days"
                }
            
            if progress_callback:
                progress_callback(30, f"Found {len(feedback_data)} training samples...")
            
            # Use the existing incremental training pipeline
            from model_training.regressions.regression_enhancement import IncrementalTrainingPipeline
            
            pipeline = IncrementalTrainingPipeline(
                model_path=None,  # Use latest model from config
                db_name=self.get_active_db(),
                verbose=True
            )
            
            if progress_callback:
                progress_callback(50, "Running incremental training on feedback data...")
            
            # Calculate date range for the pipeline
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Run incremental training
            results = pipeline.run_incremental_training_pipeline(
                date_from=start_date,
                date_to=end_date,
                limit=len(feedback_data)
            )
            
            if progress_callback:
                progress_callback(90, "Processing training results...")
            
            # Format results similar to existing incremental training
            success = results.get("status") == "success"
            
            formatted_results = {
                "success": success,
                "message": f"Weekly delta training completed on {len(feedback_data)} samples",
                "feedback_samples": len(feedback_data),
                "training_results": results,
                "days_analyzed": days_back
            }
            
            if progress_callback:
                progress_callback(100, "Weekly delta training completed\!")
            
            return formatted_results
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Weekly delta training failed: {str(e)}"
            }
    
    def show_prediction_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Basic bias detection and analytics by track/weather/distance"""
        try:
            if not self.prediction_storage:
                return {
                    "success": False,
                    "error": "Prediction storage not initialized"
                }
            
            # Get comprehensive analytics
            bias_analysis = self.prediction_storage.analyze_model_bias(days)
            performance_metrics = self.prediction_storage.get_recent_performance(days)
            confidence_calibration = self.prediction_storage.get_model_confidence_calibration(days)
            
            # Analyze performance by different contexts
            contexts_to_analyze = [
                ('track_condition', ['Good', 'Soft', 'Heavy', 'Hard']),
                ('weather', ['Clear', 'Cloudy', 'Rain', 'Windy']),
                ('race_type', ['PLAT', 'OBSTACLE', 'TROT'])
            ]
            
            context_performance = {}
            for context_type, values in contexts_to_analyze:
                context_performance[context_type] = {}
                for value in values:
                    try:
                        context_df = self.prediction_storage.get_predictions_by_context(
                            **{context_type: value}, days=days
                        )
                        if not context_df.empty and 'actual_position' in context_df.columns:
                            completed_preds = context_df[context_df['actual_position'].notna()]
                            if len(completed_preds) > 0:
                                mae = np.mean(np.abs(completed_preds['ensemble_prediction'] - completed_preds['actual_position']))
                                accuracy = np.mean(completed_preds['ensemble_prediction'] <= 3)
                                context_performance[context_type][value] = {
                                    'mae': round(mae, 3),
                                    'top3_accuracy': round(accuracy, 3),
                                    'sample_size': len(completed_preds)
                                }
                    except:
                        continue
            
            return {
                "success": True,
                "bias_analysis": bias_analysis,
                "performance_metrics": performance_metrics,
                "confidence_calibration": confidence_calibration,
                "context_performance": context_performance,
                "message": f"Prediction analytics for last {days} days completed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get prediction analytics: {str(e)}"
            }
    
    def update_results_from_database(self, progress_callback=None) -> Dict[str, Any]:
        """Update prediction storage with actual results from completed races"""
        try:
            if not self.prediction_storage:
                return {
                    "success": False,
                    "error": "Prediction storage not initialized"
                }
            
            if progress_callback:
                progress_callback(10, "Getting races with results...")
            
            # Get recent races that have results
            from core.connectors.api_daily_sync import RaceFetcher
            race_fetcher = RaceFetcher()
            
            # Get recent completed races (last 30 days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            completed_races = []
            try:
                # Fallback: get races and check for results
                daily_races = self.get_daily_races()
                completed_races = [race for race in daily_races if race.get('has_results', 0) == 1]
            except Exception:
                return {
                    "success": False,
                    "error": "Could not retrieve completed races"
                }
            
            if progress_callback:
                progress_callback(30, f"Found {len(completed_races)} completed races...")
            
            updated_count = 0
            error_count = 0
            
            for i, race in enumerate(completed_races):
                try:
                    if progress_callback:
                        progress = 30 + (i / len(completed_races)) * 60
                        progress_callback(int(progress), f"Updating results for race {race.get('comp', 'unknown')}...")
                    
                    race_comp = race['comp']
                    
                    # Get detailed race with results
                    detailed_race = race_fetcher.get_race_by_comp(race_comp)
                    if not detailed_race or not detailed_race.get('race_results'):
                        continue
                    
                    # Parse race results
                    race_results = detailed_race['race_results']
                    if isinstance(race_results, str):
                        try:
                            race_results = json.loads(race_results)
                        except json.JSONDecodeError:
                            continue
                    
                    # Update actual positions in prediction storage
                    if isinstance(race_results, dict):
                        self.prediction_storage.update_actual_results(race_comp, race_results)
                        updated_count += 1
                    
                except Exception as e:
                    print(f"Error updating results for race {race.get('comp', 'unknown')}: {str(e)}")
                    error_count += 1
                    continue
            
            if progress_callback:
                progress_callback(100, f"Results update completed\!")
            
            return {
                "success": True,
                "message": f"Updated results for {updated_count} races",
                "races_updated": updated_count,
                "races_processed": len(completed_races),
                "errors": error_count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to update results: {str(e)}"
            }
