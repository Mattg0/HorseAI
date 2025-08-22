import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json

from core.orchestrators.prediction_orchestrator import PredictionOrchestrator
from core.connectors.api_daily_sync import RaceFetcher
from utils.env_setup import AppConfig
from utils.predict_evaluator import EvaluationMetrics
from model_training.tabnet.tabnet_model import TabNetModel
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from core.calculators.static_feature_calculator import FeatureCalculator
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

class DailyPredictor:
    """Daily predictor for horse race predictions."""

    def __init__(self, model_path: str = None, db_name: str = None, verbose: bool = False, use_tabnet: bool = False):
        self.config = AppConfig('config.yaml')
        self.verbose = verbose
        self.use_tabnet = use_tabnet

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

        # Initialize TabNet-related components if requested
        self.tabnet_model = None
        self.tabnet_scaler = None
        self.feature_orchestrator = None
        self.tabnet_feature_columns = None
        
        if use_tabnet:
            self._initialize_tabnet_components()

        if verbose:
            model_type = "TabNet" if use_tabnet else "Standard"
            print(f"Daily Predictor initialized with DB: {db_name} ({model_type} mode)")

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

    def _initialize_tabnet_components(self):
        """Initialize TabNet model and related components."""
        from utils.env_setup import get_sqlite_dbpath
        
        try:
            # Initialize feature orchestrator for TabNet data preparation
            db_path = get_sqlite_dbpath(self.db_name)
            self.feature_orchestrator = FeatureEmbeddingOrchestrator(
                sqlite_path=db_path,
                verbose=self.verbose
            )
            
            if self.verbose:
                print("✅ TabNet components initialized successfully")
                
        except Exception as e:
            print(f"❌ Failed to initialize TabNet components: {e}")
            self.use_tabnet = False

    def load_tabnet_model(self, model_path: str = None) -> bool:
        """Load a trained TabNet model from file."""
        if not self.use_tabnet:
            print("❌ TabNet mode not enabled")
            return False
            
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
            import torch
            
            if model_path is None:
                # Try to find the latest TabNet model
                models_dir = Path("models")
                if models_dir.exists():
                    # Find latest model directory
                    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
                    if model_dirs:
                        latest_dir = max(model_dirs, key=lambda d: d.name)
                        # Look for TabNet files
                        tabnet_files = list(latest_dir.glob("**/tabnet_model.zip"))
                        if tabnet_files:
                            model_path = tabnet_files[0].parent
            
            if model_path is None:
                print("❌ No TabNet model path provided or found")
                return False
                
            model_path = Path(model_path)
            
            # Load TabNet model
            tabnet_model_file = model_path / "tabnet_model.zip"
            if tabnet_model_file.exists():
                self.tabnet_model = TabNetRegressor()
                self.tabnet_model.load_model(str(tabnet_model_file))
            else:
                # Try alternative naming
                tabnet_model_file = model_path / "tabnet_model.zip"
                if tabnet_model_file.exists():
                    self.tabnet_model = TabNetRegressor()
                    self.tabnet_model.load_model(str(tabnet_model_file))
                else:
                    print(f"❌ TabNet model file not found in {model_path}")
                    return False
            
            # Load scaler
            scaler_file = model_path / "tabnet_scaler.joblib"
            if scaler_file.exists():
                self.tabnet_scaler = joblib.load(scaler_file)
            else:
                print(f"❌ TabNet scaler not found in {model_path}")
                return False
                
            # Load feature configuration
            config_file = model_path / "tabnet_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.tabnet_feature_columns = config.get('feature_columns', [])
            else:
                print(f"❌ TabNet config not found in {model_path}")
                return False
            
            if self.verbose:
                print(f"✅ TabNet model loaded from {model_path}")
                print(f"📊 Features: {len(self.tabnet_feature_columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load TabNet model: {e}")
            return False

    def prepare_tabnet_features(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for TabNet prediction."""
        if not self.use_tabnet or self.feature_orchestrator is None:
            raise ValueError("TabNet mode not enabled or components not initialized")
            
        try:
            # Apply feature calculator for musique preprocessing
            if self.verbose:
                print("🔧 Applying FeatureCalculator preprocessing...")
            df_with_features = FeatureCalculator.calculate_all_features(race_df)
            
            # Prepare TabNet-specific features using the orchestrator
            if self.verbose:
                print("🔧 Preparing TabNet features...")
            tabnet_df = self.feature_orchestrator.prepare_tabnet_features(df_with_features, use_cache=False)
            
            return tabnet_df
            
        except Exception as e:
            print(f"❌ Failed to prepare TabNet features: {e}")
            raise

    def predict_with_tabnet(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using TabNet model."""
        if not self.use_tabnet:
            raise ValueError("TabNet mode not enabled")
            
        if self.tabnet_model is None:
            raise ValueError("TabNet model not loaded. Call load_tabnet_model() first")
            
        try:
            # Prepare features
            prepared_df = self.prepare_tabnet_features(race_df)
            
            # Select features that match training
            available_features = [col for col in self.tabnet_feature_columns if col in prepared_df.columns]
            missing_features = [col for col in self.tabnet_feature_columns if col not in prepared_df.columns]
            
            if missing_features and self.verbose:
                print(f"⚠️  Missing features: {missing_features}")
                
            if not available_features:
                raise ValueError("No matching features found for TabNet prediction")
            
            # Extract feature matrix
            X = prepared_df[available_features].values
            
            # Scale features
            if self.tabnet_scaler is not None:
                X_scaled = self.tabnet_scaler.transform(X)
            else:
                X_scaled = X
                
            # Generate predictions
            predictions = self.tabnet_model.predict(X_scaled).flatten()
            
            # Create result dataframe
            result_df = prepared_df.copy()
            result_df['predicted_position'] = predictions
            result_df['predicted_rank'] = result_df['predicted_position'].rank(method='min').astype(int)
            
            # Create predicted arrival order
            sorted_df = result_df.sort_values('predicted_position')
            predicted_arriv = sorted_df['numero'].tolist()
            result_df['predicted_arriv'] = str(predicted_arriv)
            
            if self.verbose:
                print(f"✅ TabNet predictions generated for {len(result_df)} horses")
                print(f"📊 Predicted top 3: {predicted_arriv[:3]}")
            
            return result_df
            
        except Exception as e:
            print(f"❌ TabNet prediction failed: {e}")
            raise

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

        # Generate prediction using appropriate model
        if self.use_tabnet and self.tabnet_model is not None:
            result_df = self.predict_with_tabnet(race_df)
            model_info = {
                'model_type': 'TabNet',
                'model_path': 'TabNet Model',
                'features_used': len(self.tabnet_feature_columns) if self.tabnet_feature_columns else 0
            }
        else:
            result_df = self.orchestrator.predict_single_race(race_df)
            model_info = self.orchestrator.get_model_info()

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
                'model_info': model_info
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


def predict_today(model_path: str = None, verbose: bool = True, use_tabnet: bool = False) -> Dict:
    """Predict today's races."""
    predictor = DailyPredictor(model_path=model_path, verbose=verbose, use_tabnet=use_tabnet)
    if use_tabnet:
        predictor.load_tabnet_model(model_path)
    return predictor.predict_races()


def fetch_and_predict_today(model_path: str = None, verbose: bool = True, use_tabnet: bool = False) -> Dict:
    """Complete workflow: fetch and predict today's races."""
    predictor = DailyPredictor(model_path=model_path, verbose=verbose, use_tabnet=use_tabnet)
    
    if use_tabnet:
        predictor.load_tabnet_model(model_path)

    fetch_results = predictor.fetch_races()
    prediction_results = predictor.predict_races()

    return {
        'fetch': fetch_results,
        'predictions': prediction_results
    }


def evaluate_today(verbose: bool = True, date : str = None ) -> Dict:
    """Evaluate today's predictions against results."""
    predictor = DailyPredictor(verbose=verbose)
    return predictor.evaluate_date(date)


def daily_report(date: str = None, verbose: bool = True) -> Dict:
    """Generate daily report."""
    predictor = DailyPredictor(verbose=verbose)
    return predictor.daily_report(date)


def predict_race(comp: str, model_path: str = None, verbose: bool = True, use_tabnet: bool = False) -> Dict:
    """Predict a specific race by ID."""
    predictor = DailyPredictor(model_path=model_path, verbose=verbose, use_tabnet=use_tabnet)
    if use_tabnet:
        predictor.load_tabnet_model(model_path)
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
   # print("📊 Fetching and predicting races...")
    predictor= DailyPredictor()
    prediction_results = predictor.fetch_races('2025-08-05')

    # Step 2: Evaluate predictions with formatted report
    print("\n📊 Evaluating predictions...")
    evaluation = evaluate_today('2025-08-05')

    # Step 3: Generate and print the evaluation report
    if 'summary_metrics' in evaluation:
        report = EvaluationReporter.report_summary_evaluation(evaluation)
        print(report)

        if 'quinte_analysis' in evaluation:
            quinte_report = EvaluationReporter.report_quinte_evaluation(evaluation['quinte_analysis'])
            print(quinte_report)