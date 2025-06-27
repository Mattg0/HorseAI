import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json

from core.orchestrators.prediction_orchestrator import PredictionOrchestrator
from core.connectors.api_daily_sync import RaceFetcher
from utils.env_setup import AppConfig
from utils.predict_evaluator import EvaluationReporter

class DailyPredictor:
    """Daily predictor for horse race predictions."""

    def __init__(self, model_path: str = None, db_name: str = None, verbose: bool = False):
        self.config = AppConfig('config.yaml')
        self.verbose = verbose

        # Get database configuration
        if db_name is None:
            db_name = self.config._config.base.active_db

        self.db_name = db_name

        # Initialize orchestrator only
        self.orchestrator = PredictionOrchestrator(
            model_path=model_path,
            db_name=db_name,
            verbose=verbose
        )

        # Initialize fetcher
        self.fetcher = RaceFetcher(
            db_name=db_name,
            verbose=verbose
        )

        if verbose:
            print(f"Daily Predictor initialized with DB: {db_name}")

    def fetch_races(self, date: str = None) -> Dict:
        """Fetch races from API for a specific date."""
        date = date or datetime.now().strftime("%Y-%m-%d")
        if self.verbose:
            print(f"🔄 Fetching races for {date}")

        results = self.fetcher.fetch_and_store_daily_races(date)

        if self.verbose:
            total = results.get('total_races', 0)
            successful = results.get('successful', 0)
            print(f"✅ Fetched {successful}/{total} races")

        return results

    def predict_races(self, date: str = None, skip_existing: bool = True) -> Dict:
        """Generate predictions for all races on a date."""
        date = date or datetime.now().strftime("%Y-%m-%d")
        races = self.fetcher.get_races_by_date(date)

        if not races:
            return {'status': 'no_races', 'date': date}

        results = []
        counts = {'predicted': 0, 'skipped': 0}

        for race in races:
            comp = race['comp']

            if skip_existing and race.get('has_predictions', 0) == 1:
                results.append({'comp': comp, 'status': 'skipped'})
                counts['skipped'] += 1
                continue

            prediction = self._predict_single_race(comp, race)
            results.append(prediction)
            if prediction['status'] == 'success':
                counts['predicted'] += 1

        return {
            'status': 'success',
            'date': date,
            'total_races': len(races),
            **counts,
            'results': results
        }

    def evaluate_race(self, comp: str) -> Dict:
        """Evaluate predictions against actual results for a race."""
        return self.orchestrator.evaluate_predictions(comp)

    def evaluate_date(self, date: str = None) -> Dict:
        """Evaluate all predictions for a specific date."""
        date = date or datetime.now().strftime("%Y-%m-%d")

        # Use orchestrator's method which returns the correct structure
        return self.orchestrator.evaluate_predictions_by_date(date)

    def daily_report(self, date: str = None) -> Dict:
        """Generate daily report of prediction performance."""
        date = date or datetime.now().strftime("%Y-%m-%d")
        races = self.fetcher.get_races_by_date(date)

        if not races:
            return {'status': 'no_data', 'date': date}

        total = len(races)
        with_predictions = sum(1 for r in races if r.get('has_predictions', 0) == 1)
        with_results = sum(1 for r in races if r.get('has_results', 0) == 1)

        return {
            'date': date,
            'total_races': total,
            'races_with_predictions': with_predictions,
            'races_with_results': with_results,
            'prediction_coverage': with_predictions / total if total > 0 else 0,
            'races': [{
                'comp': r['comp'],
                'hippo': r.get('hippo'),
                'prix': r.get('prix'),
                'has_predictions': bool(r.get('has_predictions', 0)),
                'has_results': bool(r.get('has_results', 0))
            } for r in races]
        }

    def _predict_single_race(self, comp: str, race_data: Dict) -> Dict:
        """Generate prediction for a single race."""
        participants = race_data.get('participants')
        if not participants:
            return {'comp': comp, 'status': 'error', 'error': 'No participants'}

        if isinstance(participants, str):
            participants = json.loads(participants)

        race_df = pd.DataFrame(participants)

        # Add race context attributes
        race_attrs = ['typec', 'dist', 'natpis', 'meteo', 'temperature',
                      'forceVent', 'directionVent', 'corde', 'jour',
                      'hippo', 'quinte', 'pistegp']

        for attr in race_attrs:
            if attr in race_data and race_data[attr] is not None:
                race_df[attr] = race_data[attr]

        race_df['comp'] = comp

        # Generate prediction
        result_df = self.orchestrator.predict_single_race(race_df)

        # Prepare output
        output_cols = ['numero', 'cheval', 'predicted_position', 'predicted_rank']
        optional_cols = ['cotedirect', 'jockey', 'idJockey', 'idche']
        output_cols.extend([col for col in optional_cols if col in result_df.columns])

        predictions = result_df[output_cols].to_dict('records')
        predicted_arriv = result_df['predicted_arriv'].iloc[0]

        # Store in database
        self.fetcher.update_prediction_results(comp, json.dumps({
            'metadata': {
                'race_id': comp,
                'prediction_time': datetime.now().isoformat(),
                'model_info': self.orchestrator.get_model_info()
            },
            'predictions': predictions,
            'predicted_arriv': predicted_arriv
        }))

        return {
            'comp': comp,
            'status': 'success',
            'predictions': predictions,
            'predicted_arriv': predicted_arriv
        }



# Simple IDE functions
def fetch_today(verbose: bool = True) -> Dict:
    """Fetch today's races from API."""
    predictor = DailyPredictor(verbose=verbose)
    return predictor.fetch_races()


def predict_today(model_path: str = None, verbose: bool = True) -> Dict:
    """Predict today's races."""
    predictor = DailyPredictor(model_path=model_path, verbose=verbose)
    return predictor.predict_races()


def fetch_and_predict_today(model_path: str = None, verbose: bool = True) -> Dict:
    """Complete workflow: fetch and predict today's races."""
    predictor = DailyPredictor(model_path=model_path, verbose=verbose)

    fetch_results = predictor.fetch_races()
    prediction_results = predictor.predict_races()

    return {
        'fetch': fetch_results,
        'predictions': prediction_results
    }


def evaluate_today(verbose: bool = True) -> Dict:
    """Evaluate today's predictions against results."""
    predictor = DailyPredictor(verbose=verbose)
    return predictor.evaluate_date()


def daily_report(date: str = None, verbose: bool = True) -> Dict:
    """Generate daily report."""
    predictor = DailyPredictor(verbose=verbose)
    return predictor.daily_report(date)


def predict_race(comp: str, model_path: str = None, verbose: bool = True) -> Dict:
    """Predict a specific race by ID."""
    predictor = DailyPredictor(model_path=model_path, verbose=verbose)
    return predictor.orchestrator.predict_race(comp)


def evaluate_race(comp: str, verbose: bool = True) -> Dict:
    """Evaluate a specific race prediction."""
    predictor = DailyPredictor(verbose=verbose)
    return predictor.evaluate_race()

def report_evaluation_results(self, results):
    return EvaluationReporter.report_evaluation_results(results)

def generate_evaluation_report(self, evaluation_results: Dict) -> str:
    """Generate a formatted evaluation report."""
    # For a single race
    if 'metrics' in evaluation_results:
        return EvaluationReporter.report_evaluation_results(evaluation_results)

    # For a summary
    if 'summary_metrics' in evaluation_results:
        report = EvaluationReporter.report_summary_evaluation(evaluation_results)

        # Add quinte analysis if available
        if 'quinte_analysis' in evaluation_results:
            report += "\n" + EvaluationReporter.report_quinte_evaluation(evaluation_results['quinte_analysis'])

        return report

    return "No evaluation data available"


# Example workflow
if __name__ == "__main__":
    # Step 1: Fetch and predict
    print("📊 Fetching and predicting races...")
    prediction_results = fetch_and_predict_today(verbose=True)

    # Step 2: Evaluate predictions with formatted report
    print("\n📊 Evaluating predictions...")
    evaluation = evaluate_today(verbose=True)

    # Step 3: Generate and print the evaluation report
    if 'summary_metrics' in evaluation:
        report = EvaluationReporter.report_summary_evaluation(evaluation)
        print(report)

        if 'quinte_analysis' in evaluation:
            quinte_report = EvaluationReporter.report_quinte_evaluation(evaluation['quinte_analysis'])
            print(quinte_report)