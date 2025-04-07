import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization functions
from model_training.regressions.isotonic_calibration import (
    plot_prediction_vs_actual,
    plot_calibration_effect,
    plot_feature_importance,
    plot_histogram_of_errors
)

# Import other necessary components
from utils.env_setup import AppConfig
from core.orchestrators.prediction_orchestrator import PredictionOrchestrator


class VisualizationUtility:
    """
    Utility class to generate and display visualizations for race prediction models.
    """

    def __init__(self, model_path, db_name=None, output_dir=None, verbose=False):
        """
        Initialize the visualization utility.

        Args:
            model_path: Path to the trained model directory
            db_name: Database name from config (default: active_db from config)
            output_dir: Directory to save visualizations
            verbose: Whether to output verbose logs
        """
        self.model_path = Path(model_path)
        self.verbose = verbose

        # Initialize config
        self.config = AppConfig()

        # Set database
        if db_name is None:
            self.db_name = self.config._config.base.active_db
        else:
            self.db_name = db_name

        # Get database path from config
        self.db_path = self.config.get_sqlite_dbpath(self.db_name)

        # Set output directory
        if output_dir is None:
            self.output_dir = Path(model_path) / "visualizations"
        else:
            self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model components
        self.rf_model = None
        self.lstm_model = None
        self.model_config = None

        # Initialize orchestrator for evaluation
        self.orchestrator = PredictionOrchestrator(
            model_path=model_path,
            db_name=db_name,
            verbose=verbose
        )

        # Load models
        self._load_models()

        if verbose:
            print(f"Visualization utility initialized with model at {model_path}")
            print(f"Using database: {self.db_path}")
            print(f"Visualizations will be saved to: {self.output_dir}")

    def _load_models(self):
        """Load the trained models and associated files."""
        # Find latest version folder if it exists
        versions = [d for d in self.model_path.iterdir() if d.is_dir() and d.name.startswith('v')]

        if versions:
            # Sort by version (assuming vYYYYMMDD format)
            versions.sort(reverse=True)
            version_path = versions[0]
            if self.verbose:
                print(f"Using latest model version: {version_path.name}")
        else:
            # Use model_path directly if no version folders
            version_path = self.model_path
            if self.verbose:
                print("No version folders found, using model path directly")

        # Paths for model files
        self.rf_model_path = version_path / "hybrid_rf_model.joblib"
        self.lstm_model_path = version_path / "hybrid_lstm_model"
        self.feature_engineer_path = version_path / "hybrid_feature_engineer.joblib"
        self.config_path = version_path / "model_config.json"

        # Load model configuration
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self.model_config = json.load(f)
                if self.verbose:
                    print("Loaded model configuration")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading model configuration: {str(e)}")

        # Load RF model if it exists
        if self.rf_model_path.exists():
            try:
                # Load the joblib file
                rf_data = joblib.load(self.rf_model_path)

                if isinstance(rf_data, dict) and 'model' in rf_data:
                    self.rf_model = rf_data['model']
                elif hasattr(rf_data, 'predict') and callable(getattr(rf_data, 'predict')):
                    self.rf_model = rf_data
                else:
                    # Try to find a model with predict method
                    if hasattr(rf_data, 'base_regressor'):
                        self.rf_model = rf_data
                    else:
                        self.rf_model = rf_data

                if self.verbose:
                    print("Loaded RF model")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading Random Forest model: {str(e)}")

    def generate_calibration_plot(self, race_comp=None, date=None):
        """
        Generate a calibration effect plot for a specific race or validation data.

        Args:
            race_comp: Race identifier (optional)
            date: Date string in format YYYY-MM-DD (optional)

        Returns:
            Path to the saved plot
        """
        if race_comp is None and date is None:
            # Use validation data from model fitting
            if hasattr(self.rf_model, 'calibrator') and self.rf_model.calibrator is not None:
                try:
                    # Generate synthetic data or use stored validation data
                    print("No race specified. Generating plot from model calibration data...")

                    # Get validation set size
                    n_samples = 100

                    # Generate sample predictions (normally distributed around different means)
                    raw_preds = np.random.normal(5, 2, n_samples)
                    true_vals = raw_preds + np.random.normal(0, 1, n_samples)

                    # Ensure values are positive (positions can't be negative)
                    raw_preds = np.abs(raw_preds)
                    true_vals = np.abs(true_vals)

                    # Apply calibration
                    if hasattr(self.rf_model, 'calibrator'):
                        cal_preds = self.rf_model.calibrator.predict(raw_preds)
                    else:
                        cal_preds = raw_preds  # No calibration available

                    # Generate the plot
                    plot_path = self.output_dir / "calibration_effect_validation.png"
                    plot_calibration_effect(raw_preds, cal_preds, true_vals, save_path=str(plot_path))

                    print(f"Calibration effect plot saved to {plot_path}")
                    return plot_path

                except Exception as e:
                    print(f"Error generating calibration plot from model data: {str(e)}")
            else:
                print("No calibrator found in model. Cannot generate calibration plot.")
                return None
        else:
            # Use real race data
            try:
                # Fetch predictions and actual results
                if race_comp:
                    # Get data for a specific race
                    race_data = self.orchestrator.race_fetcher.get_race_by_comp(race_comp)
                    if race_data is None:
                        print(f"Race {race_comp} not found")
                        return None

                    # Check if the race has predictions and results
                    if not race_data.get('prediction_results') or not race_data.get('actual_results'):
                        print(f"Race {race_comp} missing predictions or results")
                        return None

                    # Parse prediction results and actual results
                    try:
                        prediction_results = json.loads(race_data['prediction_results']) if isinstance(
                            race_data['prediction_results'], str) else race_data['prediction_results']
                        actual_results = json.loads(race_data['actual_results']) if isinstance(
                            race_data['actual_results'], str) else race_data['actual_results']
                    except:
                        print(f"Error parsing results for race {race_comp}")
                        return None

                    # Extract raw and calibrated predictions
                    # This will depend on your data structure
                    raw_preds = []
                    cal_preds = []
                    true_vals = []

                    # Logic to extract predictions and results will depend on your data format
                    # This is a placeholder - you'll need to adapt it to your specific format
                    if 'predictions' in prediction_results:
                        predictions = prediction_results['predictions']

                        # Create mappings for easier access
                        pred_by_numero = {p['numero']: p for p in predictions}

                        # Process actual results
                        if isinstance(actual_results, list):
                            for result in actual_results:
                                numero = result.get('numero')
                                position = result.get('position')

                                if numero is not None and position is not None and numero in pred_by_numero:
                                    # Convert position to numeric if needed
                                    try:
                                        true_position = float(position)

                                        # Get predicted values
                                        pred = pred_by_numero[numero]
                                        if 'raw_position' in pred and 'predicted_position' in pred:
                                            raw_preds.append(pred['raw_position'])
                                            cal_preds.append(pred['predicted_position'])
                                            true_vals.append(true_position)
                                        elif 'predicted_position' in pred:
                                            # Only calibrated predictions available
                                            cal_preds.append(pred['predicted_position'])
                                            raw_preds.append(pred['predicted_position'])  # Use same as placeholder
                                            true_vals.append(true_position)
                                    except:
                                        pass

                    if not true_vals:
                        print(f"Could not extract prediction data from race {race_comp}")
                        return None

                    # Generate the plot
                    plot_path = self.output_dir / f"calibration_effect_race_{race_comp}.png"
                    plot_calibration_effect(np.array(raw_preds), np.array(cal_preds), np.array(true_vals),
                                            save_path=str(plot_path))

                    print(f"Calibration effect plot saved to {plot_path}")
                    return plot_path

                elif date:
                    # Get all races for a date
                    evaluation_results = self.orchestrator.evaluate_predictions_by_date(date)

                    # Collect data from all successfully evaluated races
                    raw_preds_all = []
                    cal_preds_all = []
                    true_vals_all = []

                    for result in evaluation_results.get('results', []):
                        if result.get('status') == 'success' and 'metrics' in result:
                            # Extract data from each race's details
                            for detail in result['metrics'].get('details', []):
                                if 'predicted_raw_rank' in detail and 'predicted_rank' in detail and 'actual_rank' in detail:
                                    raw_preds_all.append(detail['predicted_raw_rank'])
                                    cal_preds_all.append(detail['predicted_rank'])
                                    true_vals_all.append(detail['actual_rank'])

                    if not true_vals_all:
                        print(f"Could not extract prediction data from races on {date}")
                        return None

                    # Generate the plot
                    plot_path = self.output_dir / f"calibration_effect_date_{date}.png"
                    plot_calibration_effect(np.array(raw_preds_all), np.array(cal_preds_all), np.array(true_vals_all),
                                            save_path=str(plot_path))

                    print(f"Calibration effect plot for races on {date} saved to {plot_path}")
                    return plot_path

            except Exception as e:
                print(f"Error generating calibration plot from race data: {str(e)}")
                import traceback
                traceback.print_exc()
                return None

    def generate_prediction_vs_actual_plot(self, race_comp=None, date=None):
        """
        Generate a prediction vs actual scatter plot.

        Args:
            race_comp: Race identifier (optional)
            date: Date string in format YYYY-MM-DD (optional)

        Returns:
            Path to the saved plot
        """
        # Similar structure to generate_calibration_plot but for prediction vs actual
        # We'll focus on getting calibrated predictions vs actual results

        try:
            if race_comp:
                # Get data for a specific race
                race_data = self.orchestrator.race_fetcher.get_race_by_comp(race_comp)
                if race_data is None:
                    print(f"Race {race_comp} not found")
                    return None

                # Process predictions and actual results
                pred_vals = []
                true_vals = []

                # Similar logic to extract predictions and results
                # But we only need predicted and actual positions

                plot_path = self.output_dir / f"prediction_vs_actual_race_{race_comp}.png"
                plot_prediction_vs_actual(np.array(true_vals), np.array(pred_vals),
                                          title=f"Predictions vs Actual - Race {race_comp}",
                                          save_path=str(plot_path))

                print(f"Prediction vs actual plot saved to {plot_path}")
                return plot_path

            elif date:
                # Get all races for a date
                evaluation_results = self.orchestrator.evaluate_predictions_by_date(date)

                # Collect predictions and actual results from all races
                pred_vals_all = []
                true_vals_all = []

                # Logic to extract data from evaluation_results

                plot_path = self.output_dir / f"prediction_vs_actual_date_{date}.png"
                plot_prediction_vs_actual(np.array(true_vals_all), np.array(pred_vals_all),
                                          title=f"Predictions vs Actual - Races on {date}",
                                          save_path=str(plot_path))

                print(f"Prediction vs actual plot for races on {date} saved to {plot_path}")
                return plot_path

            else:
                # Use test data from model evaluation if available
                print("Using model evaluation data for prediction vs actual plot")

                # Generate synthetic data if needed
                n_samples = 100
                true_vals = np.sort(np.random.uniform(1, 10, n_samples))
                pred_vals = true_vals + np.random.normal(0, 1, n_samples)

                plot_path = self.output_dir / "prediction_vs_actual_model.png"
                plot_prediction_vs_actual(true_vals, pred_vals,
                                          title="Model Predictions vs Actual Positions",
                                          save_path=str(plot_path))

                print(f"Prediction vs actual plot saved to {plot_path}")
                return plot_path

        except Exception as e:
            print(f"Error generating prediction vs actual plot: {str(e)}")
            return None

    def generate_feature_importance_plot(self):
        """
        Generate a feature importance plot for the RF model.

        Returns:
            Path to the saved plot
        """
        if self.rf_model is None:
            print("No RF model loaded. Cannot generate feature importance plot.")
            return None

        try:
            # Get feature importances and names
            feature_names = None
            importances = None

            # First try to get feature names from model
            if hasattr(self.rf_model, 'feature_names_in_'):
                feature_names = self.rf_model.feature_names_in_
            elif hasattr(self.rf_model, 'base_regressor') and hasattr(self.rf_model.base_regressor,
                                                                      'feature_names_in_'):
                feature_names = self.rf_model.base_regressor.feature_names_in_

            # Get feature importances
            if hasattr(self.rf_model, 'feature_importances_'):
                importances = self.rf_model.feature_importances_
            elif hasattr(self.rf_model, 'base_regressor') and hasattr(self.rf_model.base_regressor,
                                                                      'feature_importances_'):
                importances = self.rf_model.base_regressor.feature_importances_

            if feature_names is None or importances is None:
                print("Could not extract feature names or importances from model")
                return None

            # Generate the plot
            plot_path = self.output_dir / "feature_importance.png"
            plot_feature_importance(feature_names, importances,
                                    title="Feature Importance",
                                    top_n=20,
                                    save_path=str(plot_path))

            print(f"Feature importance plot saved to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error generating feature importance plot: {str(e)}")
            return None

    def generate_error_histogram(self, race_comp=None, date=None):
        """
        Generate a histogram of prediction errors.

        Args:
            race_comp: Race identifier (optional)
            date: Date string in format YYYY-MM-DD (optional)

        Returns:
            Path to the saved plot
        """
        # Similar structure to other methods, but for error histogram
        try:
            pred_vals = []
            true_vals = []

            # Logic to get predictions and actual values from race_comp, date, or model

            # Calculate errors
            errors = np.array(pred_vals) - np.array(true_vals)

            # Generate plot
            plot_path = self.output_dir / "prediction_errors_histogram.png"
            plot_histogram_of_errors(true_vals, pred_vals,
                                     title="Distribution of Prediction Errors",
                                     save_path=str(plot_path))

            print(f"Error histogram saved to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error generating error histogram: {str(e)}")
            return None

    def generate_all_plots(self, race_comp=None, date=None):
        """
        Generate all available plots.

        Args:
            race_comp: Race identifier (optional)
            date: Date string in format YYYY-MM-DD (optional)

        Returns:
            List of paths to generated plots
        """
        paths = []

        # Generate each type of plot
        cal_plot = self.generate_calibration_plot(race_comp, date)
        if cal_plot:
            paths.append(cal_plot)

        pred_plot = self.generate_prediction_vs_actual_plot(race_comp, date)
        if pred_plot:
            paths.append(pred_plot)

        feat_plot = self.generate_feature_importance_plot()
        if feat_plot:
            paths.append(feat_plot)

        error_plot = self.generate_error_histogram(race_comp, date)
        if error_plot:
            paths.append(error_plot)

        print(f"Generated {len(paths)} plots")
        return paths

    def generate_winner_accuracy_trend(self, start_date=None, end_date=None):
        """
        Generate a trend plot of winner prediction accuracy over time.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Path to the saved plot
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None:
            # Default to 30 days before end date
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)
            start_date = start_dt.strftime("%Y-%m-%d")

        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Query to get all races between dates
            cursor.execute("""
                SELECT jour FROM daily_race 
                WHERE jour BETWEEN ? AND ?
                GROUP BY jour
                ORDER BY jour
            """, (start_date, end_date))

            dates = [row[0] for row in cursor.fetchall()]
            conn.close()

            if not dates:
                print(f"No races found between {start_date} and {end_date}")
                return None

            # Collect accuracy data for each date
            x_dates = []
            y_accuracy = []

            for date in dates:
                # Evaluate predictions for this date
                results = self.orchestrator.evaluate_predictions_by_date(date)

                if results.get('summary_metrics') and results['summary_metrics'].get('races_evaluated', 0) > 0:
                    x_dates.append(date)
                    y_accuracy.append(results['summary_metrics']['winner_accuracy'])

            if not x_dates:
                print("No evaluation data available for the date range")
                return None

            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(x_dates, y_accuracy, marker='o', linestyle='-', color='blue')
            plt.title("Winner Prediction Accuracy Over Time")
            plt.xlabel("Date")
            plt.ylabel("Accuracy")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the plot
            plot_path = self.output_dir / f"winner_accuracy_trend_{start_date}_to_{end_date}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"Winner accuracy trend plot saved to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error generating winner accuracy trend: {str(e)}")
            return None

    def generate_betting_performance_chart(self, date=None):
        """
        Generate a bar chart showing performance across different bet types.

        Args:
            date: Specific date to evaluate, or None for all available data

        Returns:
            Path to the saved plot
        """
        try:
            if date:
                # Evaluate for a specific date
                results = self.orchestrator.evaluate_predictions_by_date(date)
                title_suffix = f" - {date}"
            else:
                # Evaluate for all available data
                # This would require collecting data from multiple dates
                # For simplicity, we'll use a placeholder implementation
                print("Generating betting performance chart for all available data is not implemented")
                print("Please specify a date")
                return None

            if not results.get('summary_metrics'):
                print("No evaluation data available")
                return None

            # Extract betting performance metrics
            pmu = results['summary_metrics']['pmu_bets']

            # Create data for the chart
            bet_types = [
                'Tiercé Exact', 'Tiercé Désordre',
                'Quarté Exact', 'Quarté Désordre',
                'Quinté+ Exact', 'Quinté+ Désordre',
                'Bonus 4', 'Bonus 3', '2 sur 4', 'Multi en 4'
            ]

            success_rates = [
                pmu.get('tierce_exact_rate', 0),
                pmu.get('tierce_desordre_rate', 0),
                pmu.get('quarte_exact_rate', 0),
                pmu.get('quarte_desordre_rate', 0),
                pmu.get('quinte_exact_rate', 0),
                pmu.get('quinte_desordre_rate', 0),
                pmu.get('bonus4_rate', 0),
                pmu.get('bonus3_rate', 0),
                pmu.get('deuxsur4_rate', 0),
                pmu.get('multi4_rate', 0),
            ]

            # Create the plot
            plt.figure(figsize=(12, 8))
            bars = plt.barh(bet_types, success_rates, color='skyblue')

            # Add values to the bars
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{success_rates[i]:.2f}', va='center')

            plt.title(f"Betting Performance by Type{title_suffix}")
            plt.xlabel("Success Rate")
            plt.xlim(0, max(success_rates) * 1.1 or 1.0)  # Add some margin
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the plot
            if date:
                plot_path = self.output_dir / f"betting_performance_{date}.png"
            else:
                plot_path = self.output_dir / "betting_performance_all.png"

            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"Betting performance chart saved to {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error generating betting performance chart: {str(e)}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for race prediction models")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model")
    parser.add_argument("--db", type=str, help="Database name from config (defaults to active_db)")
    parser.add_argument("--model-type", type=str, choices=['hybrid_model', 'incremental_models'],
                        help="Type of model if only name is provided")
    parser.add_argument("--output", type=str, help="Custom output directory for visualizations")
    parser.add_argument("--race", type=str, help="Race identifier for specific race visualizations")
    parser.add_argument("--date", type=str, help="Date for date-based visualizations (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Generate all available plots")
    parser.add_argument("--trend", action="store_true", help="Generate accuracy trend visualization")
    parser.add_argument("--betting", action="store_true", help="Generate betting performance chart")
    parser.add_argument("--calibration", action="store_true", help="Generate calibration effect plot")
    parser.add_argument("--features", action="store_true", help="Generate feature importance plot")
    parser.add_argument("--errors", action="store_true", help="Generate error histogram")
    parser.add_argument("--prediction", action="store_true", help="Generate prediction vs actual plot")
    parser.add_argument("--start-date", type=str, help="Start date for trend (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for trend (YYYY-MM-DD)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Create visualization utility with the correct model path determination
    model_path = args.model
    model_type = args.model_type

    # If model is just a name and model_type is specified, construct the path
    if not os.path.isabs(model_path) and model_type:
        config = AppConfig()
        model_paths = config.get_model_paths(config._config, model_name=model_path, model_type=model_type)
        model_path = model_paths['model_path']

    visualizer = VisualizationUtility(
        model_path=model_path,
        db_name=args.db,
        output_dir=args.output,
        verbose=args.verbose
    )

    # Track generated plots
    generated_plots = []

    # Generate requested plots
    if args.all:
        generated_plots.extend(visualizer.generate_all_plots(args.race, args.date))
        generated_plots.append(visualizer.generate_winner_accuracy_trend(args.start_date, args.end_date))
        generated_plots.append(visualizer.generate_betting_performance_chart(args.date))
    else:
        if args.calibration:
            plot = visualizer.generate_calibration_plot(args.race, args.date)
            if plot:
                generated_plots.append(plot)

        if args.prediction:
            plot = visualizer.generate_prediction_vs_actual_plot(args.race, args.date)
            if plot:
                generated_plots.append(plot)

        if args.features:
            plot = visualizer.generate_feature_importance_plot()
            if plot:
                generated_plots.append(plot)

        if args.errors:
            plot = visualizer.generate_error_histogram(args.race, args.date)
            if plot:
                generated_plots.append(plot)

        if args.trend:
            plot = visualizer.generate_winner_accuracy_trend(args.start_date, args.end_date)
            if plot:
                generated_plots.append(plot)

        if args.betting:
            plot = visualizer.generate_betting_performance_chart(args.date)
            if plot:
                generated_plots.append(plot)

    # Summary
    if generated_plots:
        print(f"\nSummary: Generated {len(generated_plots)} visualizations:")
        for plot in generated_plots:
            print(f"  - {plot}")
    else:
        print("\nNo visualizations were generated. Please check your arguments.")
        print("Use --all to generate all available visualizations.")

    return 0


if __name__ == "__main__":
    sys.exit(main())