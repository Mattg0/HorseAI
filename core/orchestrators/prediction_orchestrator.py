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

    def __init__(self, model_path: str, db_name: str = None, model_type: str = None, verbose: bool = False):
        """
        Initialize the prediction orchestrator.

        Args:
            model_path: Path to the model or model name
            db_name: Database name from config (default: active_db from config)
            model_type: Type of model ('hybrid_model' or 'incremental_models')
            verbose: Whether to output verbose logs
        """
        # Initialize config
        self.config = AppConfig()

        # Set database
        if db_name is None:
            self.db_name = self.config._config.base.active_db
        else:
            self.db_name = db_name

        # Get base model paths with active_db consideration
        self.model_paths = self.config.get_model_paths(model_name=model_path, model_type=model_type)

        # Determine if model_path is a full path or just a model name
        model_path_obj = Path(model_path)
        if model_path_obj.exists() and model_path_obj.is_dir():
            # This is a full path to a specific model version
            self.model_path = model_path_obj
        else:
            # This is a model type, need to find latest version
            model_dir = Path(self.model_paths['model_path'])
            if model_dir.exists():
                # Get all version folders, sorted newest first
                versions = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('v')],
                                  key=lambda x: x.name, reverse=True)
                if versions:
                    self.model_path = versions[0]  # Use latest version
                    if verbose:
                        print(f"Using latest model version: {self.model_path.name}")
                else:
                    # No version folders, use model_dir itself
                    self.model_path = model_dir
            else:
                # Model directory doesn't exist, create it
                os.makedirs(model_dir, exist_ok=True)
                self.model_path = model_dir

    def _setup_logging(self):
        """Set up logging."""
        logging.basicConfig(
            level=logging.INFO if not self.verbose else logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ],
            force=True  # Reset existing loggers
        )

        self.logger = logging.getLogger("PredictionOrchestrator")
        self.logger.info(f"Logging initialized to {self.log_file}")

    def predict_race(self, comp: str, blend_weight: float = 0.7) -> Dict:
        """
        Generate predictions for a specific race.

        Args:
            comp: Race identifier
            blend_weight: Weight for RF model in blend (0-1)

        Returns:
            Dictionary with prediction results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting prediction for race {comp}")

        # Check if predictor is initialized
        if self.race_predictor is None:
            return {
                'status': 'error',
                'error': 'Race predictor not initialized',
                'comp': comp
            }

        try:
            # Fetch race data from database
            race_data = self.race_fetcher.get_race_by_comp(comp)

            if race_data is None:
                return {
                    'status': 'error',
                    'error': f'Race {comp} not found in database',
                    'comp': comp
                }

            # Check if race already has predictions
            if race_data.get('prediction_results'):
                try:
                    if isinstance(race_data['prediction_results'], str):
                        pred_results = json.loads(race_data['prediction_results'])
                    else:
                        pred_results = race_data['prediction_results']

                    if pred_results:
                        self.logger.info(f"Race {comp} already has predictions")
                        return {
                            'status': 'already_predicted',
                            'message': 'Race already has predictions',
                            'comp': comp,
                            'predictions': pred_results
                        }
                except:
                    # If we can't parse the predictions, proceed to generate new ones
                    pass

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
                try:
                    participants = json.loads(participants)
                except:
                    return {
                        'status': 'error',
                        'error': 'Could not parse participant data',
                        'comp': comp
                    }

            # Process participants into feature DataFrame
            race_df = pd.DataFrame(participants)

            # Add race information to DataFrame
            for field in ['typec', 'dist', 'natpis', 'meteo', 'temperature', 'forceVent', 'directionVent', 'corde']:
                if field in race_data and race_data[field] is not None:
                    race_df[field] = race_data[field]

            # Add comp to DataFrame
            race_df['comp'] = comp

            # Generate predictions
            result_df = self.race_predictor.predict_race(race_df, blend_weight=blend_weight)

            # Select columns for output
            output_columns = [
                'numero', 'cheval', 'predicted_position', 'predicted_rank'
            ]

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
                'model_path': self.model_path,
                'blend_weight': blend_weight,
                'hippo': race_data.get('hippo'),
                'prix': race_data.get('prix'),
                'jour': race_data.get('jour'),
                'typec': race_data.get('typec'),
                'participants_count': len(prediction_results),
                'predicted_arriv': predicted_arriv  # Add the arrival string here
            }
            # Store prediction results
            prediction_data = {
                'metadata': metadata,
                'predictions': prediction_results,
                'predicted_arriv': predicted_arriv  # Add at top level of JSON
            }

            # Update database
            self.race_fetcher.update_prediction_results(comp, json.dumps(prediction_data))

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

        except Exception as e:
            self.logger.error(f"Error predicting race {comp}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            return {
                'status': 'error',
                'error': str(e),
                'comp': comp
            }

    def predict_races_by_date(self, date: str = None, blend_weight: float = 0.7) -> Dict:
        """
        Generate predictions for all races on a given date.

        Args:
            date: Date string in format YYYY-MM-DD (default: today)
            blend_weight: Weight for RF model in blend (0-1)

        Returns:
            Dictionary with prediction results for all races
        """
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
            Dictionary with evaluation metrics
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

            self.logger.info(f"Evaluation for race {comp}: "
                             f"Winner correct: {metrics['winner_correct']}, "
                             f"Podium accuracy: {metrics['podium_accuracy']:.2f}")

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

    def _calculate_summary_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate summary metrics from a list of evaluation results.

        Args:
            results: List of evaluation result dictionaries

        Returns:
            Dictionary with summary metrics
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
                }
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

        # PMU bet type success rates
        pmu_bet_types = [
            'tierce_exact', 'tierce_desordre',
            'quarte_exact', 'quarte_desordre',
            'quinte_exact', 'quinte_desordre',
            'bonus4', 'bonus3', 'deuxsur4', 'multi4'
        ]

        pmu_bet_successes = {}
        for bet_type in pmu_bet_types:
            successes = sum(1 for r in results
                            if r['status'] == 'success' and
                            r['metrics'].get('pmu_bets', {}).get(bet_type, False))
            pmu_bet_successes[bet_type] = successes
            pmu_bet_successes[f'{bet_type}_rate'] = successes / success_count

        return {
            'races_evaluated': success_count,
            'winner_accuracy': winner_correct / success_count,
            'avg_podium_accuracy': podium_accuracy,
            'avg_mean_rank_error': mean_rank_error,
            'pmu_bets': pmu_bet_successes
        }

    def evaluate_predictions_by_date(self, date: str = None) -> Dict:
        """
        Evaluate stored predictions for all races on a given date.

        Args:
            date: Date string in format YYYY-MM-DD (default: today)

        Returns:
            Dictionary with evaluation results
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
            for race in races:
                comp = race['comp']

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

            # Calculate summary metrics using our helper function
            summary_metrics = self._calculate_summary_metrics(results)

            summary = {
                'date': date,
                'total_races': len(races),
                'evaluated': summary_metrics['races_evaluated'],
                'summary_metrics': summary_metrics,
                'results': results
            }

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
        Calculate evaluation metrics based on arrival strings.

        Args:
            predicted_arriv: Predicted arrival order (e.g., "1-5-3-2-4")
            actual_arriv: Actual arrival order

        Returns:
            Dictionary with evaluation metrics
        """
        # Parse arrival strings to lists
        pred_order = predicted_arriv.split('-')
        actual_order = actual_arriv.split('-')

        # Make sure we have valid data
        if not pred_order or not actual_order:
            return {
                'error': 'Invalid arrival strings'
            }

        # 1. Winner correctly predicted
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

        # 6. Calculate Spearman correlation if scipy is available
        rank_correlation = None
        try:
            from scipy.stats import spearmanr

            # Prepare data for correlation
            common_horses = list(common_horses)
            if len(common_horses) > 2:  # Need at least 3 points for meaningful correlation
                pred_ranks_list = [pred_ranks[h] for h in common_horses]
                actual_ranks_list = [actual_ranks[h] for h in common_horses]
                correlation, _ = spearmanr(pred_ranks_list, actual_ranks_list)
                rank_correlation = correlation
        except:
            pass

        # Format details for display
        details = []
        for horse in pred_order[:min(len(pred_order), len(actual_order))]:
            if horse in actual_ranks:
                detail = {
                    'numero': horse,
                    'predicted_rank': pred_ranks.get(horse, float('nan')),
                    'actual_rank': actual_ranks.get(horse, float('nan')),
                    'rank_error': abs(pred_ranks.get(horse, 0) - actual_ranks.get(horse, 0))
                }
                details.append(detail)

        return {
            'winner_correct': winner_correct,
            'podium_accuracy': podium_accuracy,
            'mean_rank_error': mean_rank_error,
            'exacta_correct': exacta_correct,
            'trifecta_correct': trifecta_correct,
            'rank_correlation': rank_correlation,
            'details': details
        }