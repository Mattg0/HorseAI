import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Union, Any, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import core components
from utils.env_setup import AppConfig
from core.connectors.api_daily_sync import RaceFetcher
from race_prediction.race_predict import RacePredictor
from core.storage.prediction_storage import PredictionStorage, PredictionRecord


class PredictionOrchestrator:
    """
    Orchestrates the end-to-end race prediction process:
    1. Loads races from API or database
    2. Processes races with feature engineering
    3. Applies trained models to generate predictions
    4. Stores prediction results back to the database
    5. Optionally evaluates predictions against actual results
    """

    # In PredictionOrchestrator.__init__:

    def __init__(self, model_path: str = None, db_name: str = None, verbose: bool = False, 
                 auto_storage: bool = True):
        """
        Initialize the prediction orchestrator.

        Args:
            model_path: Path to the model or model name
            db_name: Database name from config (default: active_db from config)
            verbose: Whether to output verbose logs
            auto_storage: Whether to automatically store predictions in prediction storage (default: True)
        """
        # Initialize config
        self.verbose = verbose
        self.auto_storage = auto_storage
        self.config = AppConfig()

        if db_name is None:
            db_name = self.config._config.base.active_db

        self.db_name = db_name
        # Initialize components
        self.race_fetcher = RaceFetcher(db_name=self.db_name, verbose=self.verbose)
        self.race_predictor = RacePredictor(
            model_path=model_path,  # Can be None
            db_name=db_name,
            verbose=verbose
        )
        
        # Initialize prediction storage
        self.prediction_storage = PredictionStorage(self.config)

        self._setup_logging()

        # Only show initialization message if verbose
        if self.verbose:
            self.logger.info(f"Prediction Orchestrator initialized with model: {self.race_predictor.model_path}")
    def log_info(self, message):
        """Simple logging method."""
        if self.verbose:
            print(message)

    def _setup_logging(self):
        """Set up logging with proper verbose control."""
        # Create logs directory if it doesn't exist
        log_dir = self.race_predictor.model_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure the root logger - important to set up first
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)  # Default to WARNING level

        # Clear any existing handlers to avoid duplicate messages
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

        # Set up logger for this class
        self.logger = logging.getLogger("PredictionOrchestrator")

        # Set log level based on verbose flag
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Remove any existing handlers
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)

        # Add file handler always
        log_file = log_dir / f"race_predictor_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        # Add console handler only if verbose is True
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)

    def predict_race(self, comp: str, blend_weight: float = 0.9) -> pd.DataFrame:
        """
        Generate predictions for a specific race.
        """
        start_time = datetime.now()
        self.logger.info(f"Starting prediction for race {comp}")

        # Fetch race data from database
        race_data = self.race_fetcher.get_race_by_comp(comp)

        if race_data is None:
            return {
                'status': 'error',
                'error': f'Race {comp} not found in database',
                'comp': comp
            }

        # Get participants data
        participants = race_data.get('participants')
        if not participants or participants == '[]':
            return {
                'status': 'error',
                'error': 'No valid participant data found',
                'comp': comp
            }

        # Convert to DataFrame if needed
        if isinstance(participants, str):
            participants = json.loads(participants)

        # Process participants into feature DataFrame
        race_df = pd.DataFrame(participants)

        # Add race information to each participant row
        race_attributes = ['typec', 'dist', 'natpis', 'meteo', 'temperature',
                           'forceVent', 'directionVent', 'corde', 'jour',
                           'hippo', 'quinte', 'pistegp']

        for field in race_attributes:
            if field in race_data and race_data[field] is not None:
                race_df[field] = race_data[field]
                self.logger.info(f"Added race attribute {field}={race_data[field]} to all participants")

        # Add comp to DataFrame
        race_df['comp'] = comp

        # Generate predictions using RacePredictor - ONLY ONCE!
        result_df = self.race_predictor.predict_race(race_df)

        # Select columns for output
        output_columns = ['numero', 'cheval', 'predicted_position', 'predicted_rank']
        
        # Add individual model predictions and confidence scores
        for col in ['rf_prediction', 'lstm_prediction', 'tabnet_prediction', 'ensemble_confidence_score']:
            if col in result_df.columns:
                output_columns.append(col)

            # Add optional columns if available
        for col in ['cotedirect', 'jockey', 'idJockey', 'idche']:
            if col in result_df.columns:
                output_columns.append(col)

            # Create final result DataFrame
        final_result = result_df[output_columns].copy()

            # Convert to records format
        prediction_results = final_result.to_dict(orient='records')

        predicted_arriv = result_df['predicted_arriv'].iloc[0] if 'predicted_arriv' in result_df.columns else None

            # Add to metadata
        metadata = {
            'race_id': comp,
            'prediction_time': datetime.now().isoformat(),
 #           'model_path': str(self.model_path),  # Convert Path to string here
            'blend_weight': blend_weight,
            'hippo': race_data.get('hippo'),
            'prix': race_data.get('prix'),
            'jour': race_data.get('jour'),
            'typec': race_data.get('typec'),
            'participants_count': len(prediction_results),
            'predicted_arriv': predicted_arriv
        }
            # Store prediction results
        prediction_data = {
            'metadata': metadata,
            'predictions': prediction_results,
            'predicted_arriv': predicted_arriv  # Add at top level of JSON
        }

            # Update database
        self.race_fetcher.update_prediction_results(comp, json.dumps(prediction_data))
        
        # Store predictions in prediction storage system if auto_storage is enabled
        if self.auto_storage:
            try:
                self._store_race_predictions(comp, race_data, result_df, metadata)
                self.logger.info(f"Auto-stored predictions for race {comp} in prediction storage")
            except Exception as e:
                self.logger.warning(f"Failed to auto-store predictions in prediction storage: {e}")
        else:
            self.logger.debug(f"Auto-storage disabled, skipping prediction storage for race {comp}")

            # Calculate elapsed time
        elapsed_time = (datetime.now() - start_time).total_seconds()

        self.logger.info(f"Successfully predicted race {comp} in {elapsed_time:.2f} seconds")

        return {
            'status': 'success',
            'comp': comp,
            'predictions': prediction_results,
            'metadata': metadata,
            'elapsed_time': elapsed_time
        }

    def predict_races_by_date(self, date: str = None, blend_weight: float = 0.7) -> Dict:
        # Make sure logger exists
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger("PredictionOrchestrator")

        # Use today's date if none provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        self.logger.info(f"Starting predictions for races on {date}")

        try:
            # Get all races for the date
            races = self.race_fetcher.get_races_by_date(date)

            if not races:
                self.logger.info(f"No races found for {date}")
                return {
                    'status': 'no_races',
                    'message': f'No races found for {date}',
                    'date': date
                }

            self.logger.info(f"Found {len(races)} races for {date}")

            # Process each race
            results = []
            for race in races:
                comp = race['comp']

                # Skip races that already have predictions
                if race.get('has_predictions', 0) == 1:
                    self.logger.info(f"Race {comp} already has predictions, skipping")
                    results.append({
                        'status': 'already_predicted',
                        'comp': comp,
                        'message': 'Race already has predictions'
                    })
                    continue

                # Predict the race
                prediction_result = self.predict_race(comp, blend_weight=blend_weight)
                results.append(prediction_result)

            # Generate summary
            success_count = sum(1 for r in results if r['status'] == 'success')
            error_count = sum(1 for r in results if r['status'] == 'error')
            skip_count = sum(1 for r in results if r['status'] in ['already_predicted', 'no_data'])

            summary = {
                'date': date,
                'total_races': len(races),
                'predicted': success_count,
                'errors': error_count,
                'skipped': skip_count,
                'results': results
            }

            self.logger.info(f"Completed predictions for {date}: "
                             f"{success_count} successful, {error_count} errors, {skip_count} skipped")

            return summary

        except Exception as e:
            self.logger.error(f"Error processing races for {date}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            return {
                'status': 'error',
                'error': str(e),
                'date': date
            }

    def predict_single_race(self, race_df: pd.DataFrame, blend_weight: float = 0.7) -> pd.DataFrame:
        """
        Generate predictions for a single race from DataFrame.

        Args:
            race_df: DataFrame with race and participant data
            blend_weight: Weight for RF model in blend (0-1)

        Returns:
            DataFrame with predictions added
        """
        # Use the race predictor to generate predictions
        return self.race_predictor.predict_race(race_df)

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return self.race_predictor.get_model_info()
    def fetch_and_predict_races(self, date: str = None, blend_weight: float = 0.7) -> Dict:
        """
        Fetch races from API, store them, and generate predictions.

        Args:
            date: Date string in format YYYY-MM-DD (default: today)
            blend_weight: Weight for RF model in blend (0-1)

        Returns:
            Dictionary with fetch and prediction results
        """
        # Use today's date if none provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        self.logger.info(f"Starting fetch and predict for races on {date}")

        try:
            # Fetch races from API
            fetch_results = self.race_fetcher.fetch_and_store_daily_races(date)

            if fetch_results.get('status') == 'error':
                return {
                    'status': 'fetch_error',
                    'error': fetch_results.get('error'),
                    'date': date
                }

            # Extract successful races
            successful_races = [r for r in fetch_results.get('races', [])
                                if r.get('status') == 'success']

            self.logger.info(f"Successfully fetched {len(successful_races)} races for {date}")

            # Predict races
            prediction_results = self.predict_races_by_date(date, blend_weight=blend_weight)

            # Combine results
            combined_results = {
                'date': date,
                'fetch_results': fetch_results,
                'prediction_results': prediction_results
            }

            self.logger.info(f"Completed fetch and predict for {date}")

            return combined_results

        except Exception as e:
            self.logger.error(f"Error in fetch and predict for {date}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            return {
                'status': 'error',
                'error': str(e),
                'date': date
            }

    def evaluate_predictions(self, comp: str) -> Dict:
        """
        Evaluate already stored predictions against actual results for a race.
        Does NOT attempt to make new predictions.

        Args:
            comp: Race identifier

        Returns:
            Dictionary with evaluation metrics and bet results
        """
        self.logger.info(f"Evaluating stored predictions for race {comp}")

        try:
            # Get race data directly from database without using the predictor
            race_data = self.race_fetcher.get_race_by_comp(comp)

            if race_data is None:
                return {
                    'status': 'error',
                    'error': f'Race {comp} not found in database',
                    'comp': comp
                }

            # Check if race has predictions
            prediction_results = race_data.get('prediction_results')
            if not prediction_results:
                return {
                    'status': 'error',
                    'error': f'Race {comp} has no predictions',
                    'comp': comp
                }

            # Parse prediction results
            if isinstance(prediction_results, str):
                try:
                    prediction_results = json.loads(prediction_results)
                except:
                    return {
                        'status': 'error',
                        'error': 'Could not parse prediction results',
                        'comp': comp
                    }

            # Check if race has actual results
            actual_results = race_data.get('actual_results')
            if not actual_results or actual_results == 'pending':
                return {
                    'status': 'pending',
                    'message': f'Race {comp} results are still pending',
                    'comp': comp
                }

            # Parse actual results
            actual_arriv = None
            if isinstance(actual_results, str):
                try:
                    actual_results = json.loads(actual_results)
                except:
                    # It might be a direct string representation of results
                    if '-' in actual_results:
                        # Looks like a direct arrival string
                        actual_arriv = actual_results
                    else:
                        return {
                            'status': 'error',
                            'error': 'Could not parse actual results',
                            'comp': comp
                        }

            # Extract predicted_arriv from prediction results
            predicted_arriv = None
            metadata = None

            # Check different possible structures of prediction_results
            if isinstance(prediction_results, dict):
                if 'metadata' in prediction_results:
                    metadata = prediction_results['metadata']
                    predicted_arriv = metadata.get('predicted_arriv')
                elif 'predicted_arriv' in prediction_results:
                    predicted_arriv = prediction_results['predicted_arriv']

                # If we have predictions list, extract detailed predictions
                predictions = prediction_results.get('predictions', [])
            else:
                # Assume it's a list of predictions
                predictions = prediction_results
                predicted_arriv = None

            # If we don't have predicted_arriv but have predictions, construct it
            if not predicted_arriv and predictions:
                # Sort predictions by predicted_rank or predicted_position
                if isinstance(predictions[0], dict):
                    if 'predicted_rank' in predictions[0]:
                        sorted_preds = sorted(predictions, key=lambda x: x['predicted_rank'])
                    elif 'predicted_position' in predictions[0]:
                        sorted_preds = sorted(predictions, key=lambda x: x['predicted_position'])
                    else:
                        # No ranking information, use as is
                        sorted_preds = predictions

                    # Construct predicted_arriv
                    predicted_arriv = '-'.join([str(p['numero']) for p in sorted_preds])
                    self.logger.info(f"Constructed predicted_arriv: {predicted_arriv}")

            # No valid prediction format found
            if not predicted_arriv:
                return {
                    'status': 'error',
                    'error': 'Could not extract predicted arrival order',
                    'comp': comp
                }

            # Extract actual_arriv from actual results
            actual_arriv = None

            # Handle different formats of actual_results
            if isinstance(actual_results, list):
                # Sort by position and construct arrival string
                try:
                    sorted_results = sorted(
                        [r for r in actual_results if 'numero' in r and 'position' in r],
                        key=lambda x: int(x['position']) if str(x['position']).isdigit() else float('inf')
                    )
                    actual_arriv = '-'.join([str(r['numero']) for r in sorted_results])
                except Exception as e:
                    self.logger.error(f"Error sorting actual results: {str(e)}")
                    # Try to recover by using unsorted results
                    actual_arriv = '-'.join([str(r.get('numero', '')) for r in actual_results
                                             if 'numero' in r])
            elif isinstance(actual_results, str) and '-' in actual_results:
                # Direct arrival string
                actual_arriv = actual_results
            elif isinstance(actual_results, dict):
                if 'arrivee' in actual_results:
                    # Format sometimes returned from API
                    actual_arriv = actual_results['arrivee']
                elif 'ordre_arrivee' in actual_results:
                    # Another possible format
                    try:
                        ordre = json.loads(actual_results['ordre_arrivee']) if isinstance(
                            actual_results['ordre_arrivee'], str) else actual_results['ordre_arrivee']
                        sorted_results = sorted(ordre, key=lambda x: int(x['narrivee']) if str(
                            x['narrivee']).isdigit() else float('inf'))
                        actual_arriv = '-'.join([str(r.get('cheval', r.get('numero', ''))) for r in sorted_results])
                    except Exception as e:
                        self.logger.error(f"Error parsing ordre_arrivee: {str(e)}")

            if not actual_arriv:
                return {
                    'status': 'error',
                    'error': 'Could not extract actual arrival order',
                    'comp': comp
                }

            self.logger.info(f"Actual arrival: {actual_arriv}")
            self.logger.info(f"Predicted arrival: {predicted_arriv}")

            # Calculate metrics using arrival strings
            metrics = self._calculate_arriv_metrics(predicted_arriv, actual_arriv)

            # Add race info
            metrics['race_info'] = {
                'comp': comp,
                'hippo': race_data.get('hippo'),
                'prix': race_data.get('prix'),
                'jour': race_data.get('jour'),
                'typec': race_data.get('typec'),
                'model': metadata.get('model_path') if metadata else None
            }

            # Add arrival strings to output
            metrics['predicted_arriv'] = predicted_arriv
            metrics['actual_arriv'] = actual_arriv

            # Log winning bets if any
            winning_bets = metrics.get('winning_bets', [])
            if winning_bets:
                self.logger.info(f"Winning bets for race {comp}: {', '.join(winning_bets)}")
            else:
                self.logger.info(f"No winning bets for race {comp}")

            return {
                'status': 'success',
                'comp': comp,
                'metrics': metrics
            }

        except Exception as e:
            self.logger.error(f"Error evaluating predictions for race {comp}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            return {
                'status': 'error',
                'error': str(e),
                'comp': comp
            }

    def _calculate_quinte_analysis(self, quinte_results: List[Dict]) -> Dict:
        """
        Calculate specialized analysis metrics for quinte races.

        Args:
            quinte_results: List of evaluation results for quinte races

        Returns:
            Dictionary with quinte-specific metrics
        """
        quinte_count = len(quinte_results)

        if quinte_count == 0:
            return {
                'quinte_races': 0,
                'message': 'No quinte races found'
            }

        # Calculate basic metrics
        winner_correct = sum(1 for r in quinte_results if r['metrics']['winner_correct'])

        # Calculate bet success counts
        bet_types = [
            'tierce_exact', 'tierce_desordre',
            'quarte_exact', 'quarte_desordre',
            'quinte_exact', 'quinte_desordre',
            'bonus4', 'bonus3', 'deuxsur4', 'multi4'
        ]

        bet_successes = {}
        for bet_type in bet_types:
            successes = sum(1 for r in quinte_results if r['metrics']['pmu_bets'].get(bet_type, False))
            bet_successes[bet_type] = {
                'wins': successes,
                'rate': successes / quinte_count,
                'total_races': quinte_count
            }

        # Count races with at least one winning quinte bet (any type)
        races_with_quinte_bets = sum(
            1 for r in quinte_results if any(
                r['metrics']['pmu_bets'].get(bet, False) for bet in
                ['quinte_exact', 'quinte_desordre', 'bonus4', 'bonus3']
            )
        )

        # Count races with at least one winning bet of any type
        races_with_any_bets = sum(
            1 for r in quinte_results if any(
                r['metrics']['pmu_bets'].get(bet, False) for bet in bet_types
            )
        )

        # Count by number of bet types won per race
        bets_per_race = {}
        for r in quinte_results:
            winning_count = sum(1 for bet in bet_types if r['metrics']['pmu_bets'].get(bet, False))
            bets_per_race[winning_count] = bets_per_race.get(winning_count, 0) + 1

        # Get detailed info for each race
        race_details = []
        for r in quinte_results:
            metrics = r['metrics']
            race_info = metrics.get('race_info', {})

            # Get winning bets
            winning_bets = [bet for bet in bet_types if metrics['pmu_bets'].get(bet, False)]

            race_details.append({
                'comp': race_info.get('comp', 'unknown'),
                'hippo': race_info.get('hippo', 'unknown'),
                'prix': race_info.get('prix', 'unknown'),
                'jour': race_info.get('jour', 'unknown'),
                'winner_correct': metrics['winner_correct'],
                'podium_accuracy': metrics['podium_accuracy'],
                'winning_bets': winning_bets,
                'winning_bet_count': len(winning_bets),
                'predicted_arriv': metrics.get('predicted_arriv', ''),
                'actual_arriv': metrics.get('actual_arriv', '')
            })

        # Create summary
        quinte_summary = {
            'quinte_races': quinte_count,
            'winner_accuracy': winner_correct / quinte_count,
            'quinte_bet_win_rate': races_with_quinte_bets / quinte_count,
            'any_bet_win_rate': races_with_any_bets / quinte_count,
            'bet_type_details': bet_successes,
            'bets_per_race': bets_per_race,
            'race_details': race_details
        }

        return quinte_summary
    def _calculate_summary_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate summary metrics from a list of evaluation results with enhanced PMU bet reporting.

        Args:
            results: List of evaluation result dictionaries

        Returns:
            Dictionary with summary metrics and detailed PMU bet statistics
        """
        success_count = sum(1 for r in results if r['status'] == 'success')

        if success_count == 0:
            return {
                'races_evaluated': 0,
                'winner_accuracy': 0,
                'avg_podium_accuracy': 0,
                'avg_mean_rank_error': float('nan'),
                'pmu_bets': {
                    'tierce_exact': 0, 'tierce_exact_rate': 0,
                    'tierce_desordre': 0, 'tierce_desordre_rate': 0,
                    'quarte_exact': 0, 'quarte_exact_rate': 0,
                    'quarte_desordre': 0, 'quarte_desordre_rate': 0,
                    'quinte_exact': 0, 'quinte_exact_rate': 0,
                    'quinte_desordre': 0, 'quinte_desordre_rate': 0,
                    'bonus4': 0, 'bonus4_rate': 0,
                    'bonus3': 0, 'bonus3_rate': 0,
                    'deuxsur4': 0, 'deuxsur4_rate': 0,
                    'multi4': 0, 'multi4_rate': 0
                },
                'bet_type_summary': {}
            }

        # Calculate standard metrics
        winner_correct = sum(1 for r in results
                             if r['status'] == 'success' and r['metrics']['winner_correct'])

        podium_accuracy = sum(r['metrics']['podium_accuracy'] for r in results
                              if r['status'] == 'success') / success_count

        # Handle potential NaN values in mean_rank_error
        valid_errors = [r['metrics']['mean_rank_error'] for r in results
                        if r['status'] == 'success' and not pd.isna(r['metrics']['mean_rank_error'])]

        mean_rank_error = sum(valid_errors) / len(valid_errors) if valid_errors else float('nan')

        # PMU bet type success rates and detailed statistics
        pmu_bet_types = [
            'tierce_exact', 'tierce_desordre',
            'quarte_exact', 'quarte_desordre',
            'quinte_exact', 'quinte_desordre',
            'bonus4', 'bonus3', 'deuxsur4', 'multi4'
        ]

        pmu_bet_successes = {}
        bet_type_summary = {}

        # Track successes for each bet type
        for bet_type in pmu_bet_types:
            # Count successful bets
            successes = sum(1 for r in results
                            if r['status'] == 'success' and
                            r['metrics'].get('pmu_bets', {}).get(bet_type, False))

            # Success rate as percentage
            success_rate = successes / success_count

            # Store counts and rates
            pmu_bet_successes[bet_type] = successes
            pmu_bet_successes[f'{bet_type}_rate'] = success_rate

            # Create summary with win/loss count and percentage
            bet_type_summary[bet_type] = {
                'wins': successes,
                'losses': success_count - successes,
                'success_rate': success_rate * 100,  # as percentage
                'total_races': success_count
            }

        # Calculate race-by-race statistics - which races won which bets
        races_with_wins = sum(1 for r in results
                              if r['status'] == 'success' and
                              any(r['metrics'].get('pmu_bets', {}).values()))

        races_with_no_wins = success_count - races_with_wins

        # Calculate races by number of bet types won
        races_by_bet_count = {}
        for r in results:
            if r['status'] == 'success':
                wins = sum(1 for v in r['metrics'].get('pmu_bets', {}).values() if v)
                races_by_bet_count[wins] = races_by_bet_count.get(wins, 0) + 1

        bet_statistics = {
            'races_with_wins': races_with_wins,
            'races_with_no_wins': races_with_no_wins,
            'win_rate': races_with_wins / success_count if success_count > 0 else 0,
            'races_by_bet_count': races_by_bet_count
        }

        return {
            'races_evaluated': success_count,
            'winner_accuracy': winner_correct / success_count,
            'avg_podium_accuracy': podium_accuracy,
            'avg_mean_rank_error': mean_rank_error,
            'pmu_bets': pmu_bet_successes,
            'bet_type_summary': bet_type_summary,
            'bet_statistics': bet_statistics
        }

    def evaluate_predictions_by_date(self, date: str = None) -> Dict:
        """
        Evaluate stored predictions for all races on a given date,
        including specialized analysis for quinte races.

        Args:
            date: Date string in format YYYY-MM-DD (default: today)

        Returns:
            Dictionary with evaluation results including quinte analysis
        """
        # Use today's date if none provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        self.logger.info(f"Evaluating predictions for races on {date}")

        try:
            # Get all races for the date
            races = self.race_fetcher.get_races_by_date(date)

            if not races:
                self.logger.info(f"No races found for {date}")
                return {
                    'status': 'no_races',
                    'message': f'No races found for {date}',
                    'date': date
                }

            self.logger.info(f"Found {len(races)} races for {date}")

            # Process each race
            results = []
            quinte_results = []  # Always track quinte races

            for race in races:
                comp = race['comp']
                is_quinte = race.get('quinte', 0) == 1

                # Skip races without predictions or results
                if race.get('has_predictions', 0) == 0:
                    self.logger.info(f"Race {comp} has no predictions, skipping")
                    continue

                if race.get('has_results', 0) == 0:
                    self.logger.info(f"Race {comp} has no results, skipping")
                    continue

                # Evaluate without trying to predict again
                evaluation_result = self.evaluate_predictions(comp)
                results.append(evaluation_result)

                # Add to quinte results if this is a quinte race
                if is_quinte and evaluation_result['status'] == 'success':
                    # Add quinte flag and race info
                    evaluation_result['is_quinte'] = True
                    quinte_results.append(evaluation_result)

            # Calculate summary metrics using our helper function
            summary_metrics = self._calculate_summary_metrics(results)

            # Always calculate quinte analysis if quinte races exist
            quinte_summary = self._calculate_quinte_analysis(quinte_results) if quinte_results else {
                'quinte_races': 0,
                'message': 'No quinte races found'
            }

            # Create the complete summary with quinte analysis always included
            summary = {
                'date': date,
                'total_races': len(races),
                'evaluated': summary_metrics['races_evaluated'],
                'summary_metrics': summary_metrics,
                'quinte_analysis': quinte_summary,
                'results': results
            }

            # Log quinte info if we have quinte races
            if quinte_results:
                self.logger.info(f"Quinte races analysis for {date}: "
                                 f"{len(quinte_results)} quinte races analyzed, "
                                 f"Quinte bet win rate: {quinte_summary.get('quinte_bet_win_rate', 0):.2f}")

            self.logger.info(f"Completed evaluation for {date}: "
                             f"{summary_metrics['races_evaluated']} races evaluated, "
                             f"Winner accuracy: {summary_metrics['winner_accuracy']:.2f}, "
                             f"Podium accuracy: {summary_metrics['avg_podium_accuracy']:.2f}")

            return summary

        except Exception as e:
            self.logger.error(f"Error evaluating races for {date}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            return {
                'status': 'error',
                'error': str(e),
                'date': date
            }

    def _calculate_arriv_metrics(self, predicted_arriv: str, actual_arriv: str) -> Dict:
        """
        Calculate evaluation metrics based on arrival strings, including PMU bet types.

        Args:
            predicted_arriv: Predicted arrival order (e.g., "1-5-3-2-4")
            actual_arriv: Actual arrival order

        Returns:
            Dictionary with evaluation metrics and PMU bet results
        """
        # Parse arrival strings to lists
        pred_order = predicted_arriv.split('-')
        actual_order = actual_arriv.split('-')

        # Make sure we have valid data
        if not pred_order or not actual_order:
            return {
                'error': 'Invalid arrival strings'
            }

        # 1. Basic metrics
        winner_correct = pred_order[0] == actual_order[0]

        # 2. Podium accuracy (top 3)
        n_podium = min(3, len(pred_order), len(actual_order))
        pred_podium = set(pred_order[:n_podium])
        actual_podium = set(actual_order[:n_podium])
        podium_overlap = len(pred_podium.intersection(actual_podium))
        podium_accuracy = podium_overlap / n_podium if n_podium > 0 else 0

        # 3. Exacta correct (top 2 in exact order)
        exacta_correct = False
        if len(pred_order) >= 2 and len(actual_order) >= 2:
            exacta_correct = pred_order[:2] == actual_order[:2]

        # 4. Trifecta correct (top 3 in exact order)
        trifecta_correct = False
        if len(pred_order) >= 3 and len(actual_order) >= 3:
            trifecta_correct = pred_order[:3] == actual_order[:3]

        # 5. Mean rank error - requires mapping between predicted and actual positions
        # Create mapping from horse number to rank
        pred_ranks = {horse: idx + 1 for idx, horse in enumerate(pred_order)}
        actual_ranks = {horse: idx + 1 for idx, horse in enumerate(actual_order)}

        # Find common horses
        common_horses = set(pred_ranks.keys()) & set(actual_ranks.keys())

        if common_horses:
            rank_errors = [abs(pred_ranks[horse] - actual_ranks[horse]) for horse in common_horses]
            mean_rank_error = sum(rank_errors) / len(rank_errors)
        else:
            mean_rank_error = float('nan')

        # 6. Calculate PMU bet type results
        pmu_bets = {}

        # 6.1 Tiercé (top 3)
        if len(pred_order) >= 3 and len(actual_order) >= 3:
            # Tiercé exact (1-2-3 in order)
            pmu_bets['tierce_exact'] = pred_order[:3] == actual_order[:3]

            # Tiercé désordre (1-2-3 in any order)
            pmu_bets['tierce_desordre'] = set(pred_order[:3]) == set(actual_order[:3])
        else:
            pmu_bets['tierce_exact'] = False
            pmu_bets['tierce_desordre'] = False

        # 6.2 Quarté (top 4)
        if len(pred_order) >= 4 and len(actual_order) >= 4:
            # Quarté exact (1-2-3-4 in order)
            pmu_bets['quarte_exact'] = pred_order[:4] == actual_order[:4]

            # Quarté désordre (1-2-3-4 in any order)
            pmu_bets['quarte_desordre'] = set(pred_order[:4]) == set(actual_order[:4])

            # Bonus 4 (first horse correct, other 3 in top 4 in any order)
            pmu_bets['bonus4'] = (pred_order[0] == actual_order[0] and
                                  len(set(pred_order[1:4]) & set(actual_order[1:4])) >= 3)
        else:
            pmu_bets['quarte_exact'] = False
            pmu_bets['quarte_desordre'] = False
            pmu_bets['bonus4'] = False

        # 6.3 Quinté+ (top 5)
        if len(pred_order) >= 5 and len(actual_order) >= 5:
            # Quinté+ exact (1-2-3-4-5 in order)
            pmu_bets['quinte_exact'] = pred_order[:5] == actual_order[:5]

            # Quinté+ désordre (1-2-3-4-5 in any order)
            pmu_bets['quinte_desordre'] = set(pred_order[:5]) == set(actual_order[:5])

            # Bonus 3 (first horse correct, other 2 in top 3 in any order)
            pmu_bets['bonus3'] = (pred_order[0] == actual_order[0] and
                                  len(set(pred_order[1:3]) & set(actual_order[1:3])) >= 2)

            # 2 sur 4 (at least 2 of the top 4 correct in any order)
            pmu_bets['deuxsur4'] = len(set(pred_order[:4]) & set(actual_order[:4])) >= 2

            # Multi en 4 (top 4 correct in any order)
            pmu_bets['multi4'] = set(pred_order[:4]) == set(actual_order[:4])
        else:
            pmu_bets['quinte_exact'] = False
            pmu_bets['quinte_desordre'] = False
            pmu_bets['bonus3'] = False
            pmu_bets['deuxsur4'] = False
            pmu_bets['multi4'] = False
        # 6.4 Extended Quinté analysis (6 and 7 horses)
        if len(pred_order) >= 6 and len(actual_order) >= 5:
            # Playing 6 horses - need 5 in top 5 (any order)
            pmu_bets['quinte_6horses'] = len(set(pred_order[:6]) & set(actual_order[:5])) == 5

            # Bonus 4 with 6 horses - first correct + 3 others in top 4
            pmu_bets['bonus4_6horses'] = (pred_order[0] == actual_order[0] and
                                          len(set(pred_order[1:6]) & set(actual_order[1:4])) >= 3)
        else:
            pmu_bets['quinte_6horses'] = False
            pmu_bets['bonus4_6horses'] = False

        if len(pred_order) >= 7 and len(actual_order) >= 5:
            # Playing 7 horses - need 5 in top 5 (any order)
            pmu_bets['quinte_7horses'] = len(set(pred_order[:7]) & set(actual_order[:5])) == 5

            # Bonus 4 with 7 horses
            pmu_bets['bonus4_7horses'] = (pred_order[0] == actual_order[0] and
                                          len(set(pred_order[1:7]) & set(actual_order[1:4])) >= 3)

            # Bonus 3 with 7 horses
            pmu_bets['bonus3_7horses'] = (pred_order[0] == actual_order[0] and
                                          len(set(pred_order[1:7]) & set(actual_order[1:3])) >= 2)
        else:
            pmu_bets['quinte_7horses'] = False
            pmu_bets['bonus4_7horses'] = False
            pmu_bets['bonus3_7horses'] = False
        # 7. Determine which bet types were won (for easier reporting)
        winning_bets = [bet_type for bet_type, result in pmu_bets.items() if result]

        return {
            'winner_correct': winner_correct,
            'podium_accuracy': podium_accuracy,
            'mean_rank_error': mean_rank_error,
            'exacta_correct': exacta_correct,
            'trifecta_correct': trifecta_correct,
            'pmu_bets': pmu_bets,
            'winning_bets': winning_bets
        }
    
    def _store_race_predictions(self, race_id: str, race_data: Dict, result_df: pd.DataFrame, metadata: Dict):
        """Store race predictions in the prediction storage system"""
        try:
            # Get model versions from race predictor
            model_versions = {}
            if hasattr(self.race_predictor, 'rf_model') and self.race_predictor.rf_model is not None:
                model_versions['rf'] = str(self.race_predictor.model_path)
            if hasattr(self.race_predictor, 'lstm_model') and self.race_predictor.lstm_model is not None:
                model_versions['lstm'] = str(self.race_predictor.model_path)
            if hasattr(self.race_predictor, 'tabnet_model') and self.race_predictor.tabnet_model is not None:
                model_versions['tabnet'] = str(self.race_predictor.model_path)
            
            # Prepare prediction records for each horse
            prediction_records = []
            timestamp = datetime.now()
            
            for _, row in result_df.iterrows():
                # Extract individual model predictions if available
                rf_pred = row.get('rf_prediction', None)
                lstm_pred = row.get('lstm_prediction', None)
                tabnet_pred = row.get('tabnet_prediction', None)
                ensemble_pred = row.get('predicted_position', None)
                
                # Calculate confidence score if available
                confidence_score = None
                if all(pred is not None for pred in [rf_pred, lstm_pred, tabnet_pred]):
                    # Simple confidence based on model agreement
                    predictions = [rf_pred, lstm_pred, tabnet_pred]
                    mean_pred = np.mean(predictions)
                    std_pred = np.std(predictions)
                    confidence_score = 1.0 / (1.0 + std_pred) if std_pred > 0 else 1.0
                
                # Create prediction record
                record = PredictionRecord(
                    race_id=race_id,
                    timestamp=timestamp,
                    horse_id=str(row.get('numero', row.get('idche', 'unknown'))),
                    rf_prediction=rf_pred,
                    lstm_prediction=lstm_pred,
                    tabnet_prediction=tabnet_pred,
                    ensemble_prediction=ensemble_pred,
                    ensemble_confidence_score=confidence_score,
                    actual_position=None,  # Will be updated later when results are available
                    distance=race_data.get('dist'),
                    track_condition=race_data.get('natpis'),
                    weather=race_data.get('meteo'),
                    field_size=len(result_df),
                    race_type=race_data.get('typec'),
                    model_versions=model_versions,
                    prediction_metadata={
                        'hippo': race_data.get('hippo'),
                        'prix': race_data.get('prix'),
                        'jour': race_data.get('jour'),
                        'quinte': race_data.get('quinte', 0),
                        'blend_weight': metadata.get('blend_weight'),
                        'horse_name': row.get('cheval', ''),
                        'jockey': row.get('jockey', ''),
                        'predicted_rank': row.get('predicted_rank', None)
                    }
                )
                prediction_records.append(record)
            
            # Store batch predictions
            record_ids = self.prediction_storage.store_batch_predictions(prediction_records)
            self.logger.info(f"Stored {len(record_ids)} prediction records for race {race_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing predictions for race {race_id}: {e}")
            raise
    
    def store_race_predictions_manually(self, race_id: str) -> Dict[str, Any]:
        """Manually store predictions for a specific race (useful when auto_storage is disabled)"""
        try:
            # Get race data and prediction results
            race_data = self.race_fetcher.get_race_by_comp(race_id)
            if not race_data:
                return {
                    "success": False,
                    "error": f"Race {race_id} not found"
                }
            
            prediction_results = race_data.get('prediction_results')
            if not prediction_results:
                return {
                    "success": False,
                    "error": f"No prediction results found for race {race_id}"
                }
            
            # Parse prediction results if they're stored as JSON string
            if isinstance(prediction_results, str):
                try:
                    prediction_results = json.loads(prediction_results)
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": "Invalid prediction results format"
                    }
            
            # Convert predictions back to DataFrame format for storage
            predictions_df = pd.DataFrame(prediction_results.get('predictions', []))
            metadata = prediction_results.get('metadata', {})
            
            # Store using existing method
            self._store_race_predictions(race_id, race_data, predictions_df, metadata)
            
            return {
                "success": True,
                "message": f"Manually stored predictions for race {race_id}",
                "race_id": race_id,
                "predictions_count": len(predictions_df)
            }
            
        except Exception as e:
            self.logger.error(f"Error manually storing predictions for race {race_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_race_results(self, race_id: str, results: Dict[str, int]):
        """Update actual race results in prediction storage"""
        try:
            self.prediction_storage.update_actual_results(race_id, results)
            self.logger.info(f"Updated actual results for race {race_id}")
        except Exception as e:
            self.logger.error(f"Error updating results for race {race_id}: {e}")
    
    def get_prediction_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get recent prediction performance metrics"""
        return self.prediction_storage.get_recent_performance(days)
    
    def analyze_prediction_bias(self, days: int = 60) -> Dict[str, Any]:
        """Analyze systematic biases in predictions"""
        return self.prediction_storage.analyze_model_bias(days)
    
    def get_training_feedback_data(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent prediction data for model training feedback"""
        return self.prediction_storage.get_training_feedback(days)
    
    def export_prediction_data(self, start_date: Optional[datetime] = None, 
                              end_date: Optional[datetime] = None) -> str:
        """Export prediction data for analysis"""
        return self.prediction_storage.export_for_analysis(start_date, end_date)
    
    def suggest_optimal_blend_weights(self, days: int = 30) -> Dict[str, float]:
        """Get suggestions for optimal model blend weights based on recent performance"""
        return self.prediction_storage.suggest_blend_weights(days)