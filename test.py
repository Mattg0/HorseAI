import os
import sys
import json
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import logging
from typing import List, Dict, Any, Tuple
import warnings

# Filter specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting behavior in `replace` is deprecated.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import core components
from utils.env_setup import AppConfig
from core.orchestrators.prediction_orchestrator import PredictionOrchestrator
from core.connectors.api_daily_sync import RaceFetcher
from race_prediction.race_predict import RacePredictor


class BlendingComparison:
    """
    Class to compare race prediction results with different blending values.
    """

    def __init__(self, model_path: str, db_name: str = None, output_dir: str = None,
                 blending_values: List[float] = None, verbose: bool = False):
        """
        Initialize the blending comparison tool.

        Args:
            model_path: Path to the model or model name
            db_name: Database name from config (default: active_db from config)
            output_dir: Directory to save result files
            blending_values: List of blending values to test (default: [0.0, 0.25, 0.5, 0.75, 1.0])
            verbose: Whether to output verbose logs
        """
        # Initialize config
        self.config = AppConfig()

        # Set database
        if db_name is None:
            self.db_name = self.config._config.base.active_db
        else:
            self.db_name = db_name

        # Get database path
        self.db_path = self.config.get_sqlite_dbpath(self.db_name)

        # Set model path
        self.model_path = model_path

        # Set blending values
        self.blending_values = blending_values or [0.0, 0.25, 0.5, 0.75, 1.0]

        # Set verbosity
        self.verbose = verbose

        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"results/blend_comparison_{timestamp}")
        else:
            self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Initialize race fetcher for database access
        self.race_fetcher = RaceFetcher(db_name=self.db_name)

        # Store results
        self.results = {}

        print(f"Blending comparison initialized:")
        print(f"  - Model path: {self.model_path}")
        print(f"  - Database: {self.db_name} ({self.db_path})")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Blending values: {self.blending_values}")
        self.logger.info(f"Blending comparison initialized:")
        self.logger.info(f"  - Model path: {self.model_path}")
        self.logger.info(f"  - Database: {self.db_name} ({self.db_path})")
        self.logger.info(f"  - Output directory: {self.output_dir}")
        self.logger.info(f"  - Blending values: {self.blending_values}")

    def _setup_logging(self):
        """Set up logging."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"blend_comparison_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True  # Reset existing loggers
        )

        self.logger = logging.getLogger("BlendingComparison")
        self.logger.info(f"Logging initialized to {log_file}")

    def _clear_predictions(self):
        """Clear prediction results from the database."""
        print("Clearing existing prediction results from database...")
        self.logger.info("Clearing existing prediction results from database...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update all daily_race entries to clear prediction_results
        cursor.execute("UPDATE daily_race SET prediction_results = NULL")

        # Commit and close
        conn.commit()
        count = cursor.rowcount
        conn.close()

        print(f"Cleared prediction results from {count} races")
        self.logger.info(f"Cleared prediction results from {count} races")

    def _get_evaluation_dates(self, start_date: str = None, num_days: int = 3) -> List[str]:
        """
        Get dates for evaluation.

        Args:
            start_date: Starting date in YYYY-MM-DD format (default: 3 days ago)
            num_days: Number of days to evaluate

        Returns:
            List of date strings in YYYY-MM-DD format
        """
        if start_date is None:
            # Default to 3 days ago
            start_dt = datetime.now() - timedelta(days=num_days)
            start_date = start_dt.strftime("%Y-%m-%d")

        # Convert to datetime
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        # Generate list of dates
        dates = [
            (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(num_days)
        ]

        return dates

    def _check_race_availability(self, dates: List[str]) -> Tuple[bool, int]:
        """
        Check if races are available for the specified dates.

        Args:
            dates: List of date strings

        Returns:
            Tuple of (all_have_results, total_race_count)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        all_have_results = True
        total_race_count = 0

        for date in dates:
            # Check if races exist for the date
            cursor.execute(
                "SELECT COUNT(*) FROM daily_race WHERE jour = ?",
                (date,)
            )
            race_count = cursor.fetchone()[0]

            # Check if they have results
            cursor.execute(
                "SELECT COUNT(*) FROM daily_race WHERE jour = ? AND actual_results IS NOT NULL AND actual_results != 'pending'",
                (date,)
            )
            results_count = cursor.fetchone()[0]

            if race_count == 0:
                print(f"Warning: No races found for date {date}")
                self.logger.warning(f"No races found for date {date}")
                all_have_results = False
            elif results_count < race_count:
                print(f"Warning: Only {results_count}/{race_count} races have results for date {date}")
                self.logger.warning(f"Only {results_count}/{race_count} races have results for date {date}")
                all_have_results = False
            else:
                print(f"Found {race_count} races with results for date {date}")
                self.logger.info(f"Found {race_count} races with results for date {date}")

            total_race_count += race_count

        conn.close()

        return all_have_results, total_race_count

    def evaluate_blending_values(self, dates: List[str] = None, num_days: int = 3) -> Dict[str, Any]:
        """
        Evaluate prediction performance for different blending values.

        Args:
            dates: List of dates to evaluate (default: last 3 days)
            num_days: Number of days to evaluate if dates not provided

        Returns:
            Dictionary with evaluation results for each blending value
        """
        # Get dates for evaluation if not provided
        if dates is None:
            dates = self._get_evaluation_dates(num_days=num_days)

        print(f"Evaluating blending values for dates: {dates}")
        self.logger.info(f"Evaluating blending values for dates: {dates}")

        # Check if races are available
        all_have_results, total_race_count = self._check_race_availability(dates)

        if total_race_count == 0:
            print("Error: No races found for the specified dates. Aborting.")
            self.logger.error("No races found for the specified dates. Aborting.")
            return {"status": "error", "message": "No races found"}

        if not all_have_results:
            print("Warning: Some races may not have results. Evaluation may be incomplete.")
            self.logger.warning("Some races may not have results. Evaluation may be incomplete.")

        # Evaluate each blending value
        for blend_value in self.blending_values:
            print(f"\n{'-' * 40}")
            print(f"Evaluating blending value: {blend_value}")
            self.logger.info(f"\n{'=' * 40}")
            self.logger.info(f"Evaluating blending value: {blend_value}")

            # Clear previous predictions
            self._clear_predictions()

            # Initialize orchestrator with the current blending value
            orchestrator = PredictionOrchestrator(
                model_path=self.model_path,
                db_name=self.db_name,
                verbose=self.verbose
            )

            # Process each date
            date_results = []
            for date in dates:
                print(f"Predicting races for date: {date}")
                self.logger.info(f"Predicting races for date: {date}")

                # Predict races for the date with current blend value
                prediction_results = orchestrator.predict_races_by_date(date, blend_weight=blend_value)

                # Evaluate predictions
                print(f"Evaluating predictions for date: {date}")
                self.logger.info(f"Evaluating predictions for date: {date}")
                evaluation_results = orchestrator.evaluate_predictions_by_date(date)

                date_results.append({
                    "date": date,
                    "prediction_results": prediction_results,
                    "evaluation_results": evaluation_results
                })

            # Store results for this blending value
            self.results[str(blend_value)] = {
                "blend_value": blend_value,
                "date_results": date_results,
                "summary": self._calculate_summary(date_results)
            }

            # Save intermediate results
            self._save_results()

        # Generate comparison report
        comparison = self._generate_comparison()

        # Visualize results
        self._visualize_results()

        return comparison

    def _calculate_summary(self, date_results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate summary metrics across all evaluated dates.

        Args:
            date_results: List of results for each date

        Returns:
            Dictionary with summary metrics
        """
        # Initialize counters and accumulators
        total_races = 0
        total_evaluated = 0
        total_correct_winner = 0
        total_podium_accuracy = 0
        total_races_with_metrics = 0  # Count races that actually have metrics

        # PMU bet types - initialize counters
        pmu_bets = {
            'tierce_exact': 0, 'tierce_desordre': 0,
            'quarte_exact': 0, 'quarte_desordre': 0,
            'quinte_exact': 0, 'quinte_desordre': 0,
            'bonus4': 0, 'bonus3': 0,
            'deuxsur4': 0, 'multi4': 0
        }

        # Process each date
        for date_result in date_results:
            eval_result = date_result.get('evaluation_results', {})

            # Get summary metrics if available
            if 'summary_metrics' in eval_result:
                metrics = eval_result['summary_metrics']

                # Add to totals
                races_evaluated = metrics.get('races_evaluated', 0)
                total_races_with_metrics += races_evaluated

                if races_evaluated > 0:
                    # Calculate proportional metrics
                    winner_accuracy = metrics.get('winner_accuracy', 0)
                    podium_accuracy = metrics.get('avg_podium_accuracy', 0)

                    total_correct_winner += winner_accuracy * races_evaluated
                    total_podium_accuracy += podium_accuracy * races_evaluated

                    # Add PMU bet counts
                    for bet_type in pmu_bets.keys():
                        pmu_bets[bet_type] += metrics.get('pmu_bets', {}).get(bet_type, 0)

            # Count total races
            pred_result = date_result.get('prediction_results', {})
            total_races += pred_result.get('total_races', 0)
            total_evaluated += pred_result.get('predicted', 0)

        # Calculate averages
        if total_races_with_metrics > 0:
            avg_winner_accuracy = total_correct_winner / total_races_with_metrics
            avg_podium_accuracy = total_podium_accuracy / total_races_with_metrics

            # Calculate PMU bet rates
            pmu_rates = {f"{k}_rate": v / total_races_with_metrics for k, v in pmu_bets.items()}

            return {
                "total_races": total_races,
                "total_evaluated": total_evaluated,
                "total_with_metrics": total_races_with_metrics,
                "winner_accuracy": avg_winner_accuracy,
                "podium_accuracy": avg_podium_accuracy,
                "pmu_bets": {**pmu_bets, **pmu_rates}
            }
        else:
            return {
                "total_races": total_races,
                "total_evaluated": total_evaluated,
                "total_with_metrics": total_races_with_metrics,
                "winner_accuracy": 0,
                "podium_accuracy": 0,
                "pmu_bets": {k: 0 for k in pmu_bets.keys()}
            }

    def _generate_comparison(self) -> Dict[str, Any]:
        """
        Generate a comparison of results for different blending values.

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "blending_values": self.blending_values,
            "metrics": {}
        }

        # Collect key metrics for each blending value
        winner_accuracy = []
        podium_accuracy = []
        tierce_exact_rate = []
        tierce_desordre_rate = []
        quarte_exact_rate = []

        # Find best values for each metric
        best_winner = {"value": 0, "blend": None}
        best_podium = {"value": 0, "blend": None}
        best_tierce_exact = {"value": 0, "blend": None}
        best_tierce_desordre = {"value": 0, "blend": None}
        best_quarte_exact = {"value": 0, "blend": None}

        for blend_str, result in self.results.items():
            blend_value = float(blend_str)
            summary = result.get('summary', {})

            # Extract key metrics
            win_acc = summary.get('winner_accuracy', 0)
            pod_acc = summary.get('podium_accuracy', 0)
            t_exact = summary.get('pmu_bets', {}).get('tierce_exact_rate', 0)
            t_desordre = summary.get('pmu_bets', {}).get('tierce_desordre_rate', 0)
            q_exact = summary.get('pmu_bets', {}).get('quarte_exact_rate', 0)

            # Store for comparison
            winner_accuracy.append(win_acc)
            podium_accuracy.append(pod_acc)
            tierce_exact_rate.append(t_exact)
            tierce_desordre_rate.append(t_desordre)
            quarte_exact_rate.append(q_exact)

            # Update best values
            if win_acc > best_winner["value"]:
                best_winner = {"value": win_acc, "blend": blend_value}
            if pod_acc > best_podium["value"]:
                best_podium = {"value": pod_acc, "blend": blend_value}
            if t_exact > best_tierce_exact["value"]:
                best_tierce_exact = {"value": t_exact, "blend": blend_value}
            if t_desordre > best_tierce_desordre["value"]:
                best_tierce_desordre = {"value": t_desordre, "blend": blend_value}
            if q_exact > best_quarte_exact["value"]:
                best_quarte_exact = {"value": q_exact, "blend": blend_value}

        # Store metric arrays
        comparison["metrics"]["winner_accuracy"] = winner_accuracy
        comparison["metrics"]["podium_accuracy"] = podium_accuracy
        comparison["metrics"]["tierce_exact_rate"] = tierce_exact_rate
        comparison["metrics"]["tierce_desordre_rate"] = tierce_desordre_rate
        comparison["metrics"]["quarte_exact_rate"] = quarte_exact_rate

        # Store best values
        comparison["best"] = {
            "winner": best_winner,
            "podium": best_podium,
            "tierce_exact": best_tierce_exact,
            "tierce_desordre": best_tierce_desordre,
            "quarte_exact": best_quarte_exact
        }

        # Calculate overall recommendation (weighted average)
        weights = {
            "winner": 0.3,
            "podium": 0.3,
            "tierce_exact": 0.2,
            "tierce_desordre": 0.1,
            "quarte_exact": 0.1
        }

        # Create a score for each blending value
        scores = {}
        for blend_str, result in self.results.items():
            blend_value = float(blend_str)
            summary = result.get('summary', {})

            # Handle zero case for best values (avoid division by zero)
            normalized_scores = []

            if best_winner["value"] > 0:
                normalized_scores.append(weights["winner"] * summary.get('winner_accuracy', 0) / best_winner["value"])

            if best_podium["value"] > 0:
                normalized_scores.append(weights["podium"] * summary.get('podium_accuracy', 0) / best_podium["value"])

            if best_tierce_exact["value"] > 0:
                normalized_scores.append(
                    weights["tierce_exact"] * summary.get('pmu_bets', {}).get('tierce_exact_rate', 0) /
                    best_tierce_exact["value"])

            if best_tierce_desordre["value"] > 0:
                normalized_scores.append(
                    weights["tierce_desordre"] * summary.get('pmu_bets', {}).get('tierce_desordre_rate', 0) /
                    best_tierce_desordre["value"])

            if best_quarte_exact["value"] > 0:
                normalized_scores.append(
                    weights["quarte_exact"] * summary.get('pmu_bets', {}).get('quarte_exact_rate', 0) /
                    best_quarte_exact["value"])

            # Calculate score as sum of normalized values
            if normalized_scores:
                scores[blend_value] = sum(normalized_scores)
            else:
                scores[blend_value] = 0

        # Find best overall blend value
        if scores:
            best_blend = max(scores.items(), key=lambda x: x[1])
        else:
            best_blend = (self.blending_values[0], 0)  # Default to first value if no scores

        comparison["recommendation"] = {
            "blend_value": best_blend[0],
            "score": best_blend[1],
            "scores": scores
        }

        # Save comparison to file
        with open(self.output_dir / "comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"Comparison generated and saved to {self.output_dir / 'comparison.json'}")
        self.logger.info(f"Comparison generated and saved to {self.output_dir / 'comparison.json'}")

        # Print recommendation
        print("\n" + "=" * 50)
        print("BLENDING COMPARISON RESULTS:")
        print(f"Best winner accuracy:     {best_winner['value']:.4f} (blend={best_winner['blend']})")
        print(f"Best podium accuracy:     {best_podium['value']:.4f} (blend={best_podium['blend']})")
        print(f"Best Tiercé (exact):      {best_tierce_exact['value']:.4f} (blend={best_tierce_exact['blend']})")
        print(f"Best Tiercé (désordre):   {best_tierce_desordre['value']:.4f} (blend={best_tierce_desordre['blend']})")
        print(f"Best Quarté (exact):      {best_quarte_exact['value']:.4f} (blend={best_quarte_exact['blend']})")
        print("\nRECOMMENDED BLENDING VALUE:")
        print(f"   {best_blend[0]:.2f} (weighted score: {best_blend[1]:.4f})")
        print("=" * 50)

        self.logger.info("\n" + "=" * 50)
        self.logger.info("BLENDING COMPARISON RESULTS:")
        self.logger.info(f"Best winner accuracy:     {best_winner['value']:.4f} (blend={best_winner['blend']})")
        self.logger.info(f"Best podium accuracy:     {best_podium['value']:.4f} (blend={best_podium['blend']})")
        self.logger.info(
            f"Best Tiercé (exact):      {best_tierce_exact['value']:.4f} (blend={best_tierce_exact['blend']})")
        self.logger.info(
            f"Best Tiercé (désordre):   {best_tierce_desordre['value']:.4f} (blend={best_tierce_desordre['blend']})")
        self.logger.info(
            f"Best Quarté (exact):      {best_quarte_exact['value']:.4f} (blend={best_quarte_exact['blend']})")
        self.logger.info("\nRECOMMENDED BLENDING VALUE:")
        self.logger.info(f"   {best_blend[0]:.2f} (weighted score: {best_blend[1]:.4f})")
        self.logger.info("=" * 50)

        return comparison

    def _save_results(self):
        """Save current results to file."""
        results_file = self.output_dir / "results.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"Results saved to {results_file}")
        self.logger.info(f"Results saved to {results_file}")

    def _visualize_results(self):
        """Create visualizations of the results."""
        visualizations_dir = self.output_dir / "visualizations"
        visualizations_dir.mkdir(exist_ok=True)

        # Extract data for plotting
        blend_values = self.blending_values
        winner_accuracy = []
        podium_accuracy = []
        tierce_exact_rate = []
        tierce_desordre_rate = []

        for blend_value in blend_values:
            blend_str = str(blend_value)
            if blend_str in self.results:
                summary = self.results[blend_str].get('summary', {})
                winner_accuracy.append(summary.get('winner_accuracy', 0))
                podium_accuracy.append(summary.get('podium_accuracy', 0))
                tierce_exact_rate.append(summary.get('pmu_bets', {}).get('tierce_exact_rate', 0))
                tierce_desordre_rate.append(summary.get('pmu_bets', {}).get('tierce_desordre_rate', 0))

        # Plot 1: Winner and Podium Accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(blend_values, winner_accuracy, 'o-', label='Winner Accuracy')
        plt.plot(blend_values, podium_accuracy, 's-', label='Podium Accuracy')
        plt.xlabel('RF Model Weight')
        plt.ylabel('Accuracy')
        plt.title('Winner and Podium Accuracy vs. RF Weight')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(blend_values)

        # Annotate values
        for i, val in enumerate(winner_accuracy):
            plt.annotate(f"{val:.3f}", (blend_values[i], val), xytext=(0, 10),
                         textcoords='offset points', ha='center')
        for i, val in enumerate(podium_accuracy):
            plt.annotate(f"{val:.3f}", (blend_values[i], val), xytext=(0, -15),
                         textcoords='offset points', ha='center')

        plt.savefig(visualizations_dir / "accuracy_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Tiercé Rates
        plt.figure(figsize=(12, 6))
        plt.plot(blend_values, tierce_exact_rate, 'o-', label='Tiercé Exact Rate')
        plt.plot(blend_values, tierce_desordre_rate, 's-', label='Tiercé Désordre Rate')
        plt.xlabel('RF Model Weight')
        plt.ylabel('Rate')
        plt.title('Tiercé Hit Rates vs. RF Weight')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(blend_values)

        # Annotate values
        for i, val in enumerate(tierce_exact_rate):
            plt.annotate(f"{val:.3f}", (blend_values[i], val), xytext=(0, 10),
                         textcoords='offset points', ha='center')
        for i, val in enumerate(tierce_desordre_rate):
            plt.annotate(f"{val:.3f}", (blend_values[i], val), xytext=(0, -15),
                         textcoords='offset points', ha='center')

        plt.savefig(visualizations_dir / "tierce_rates_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 3: Combined metrics on radar chart
        metrics_names = ['Winner Accuracy', 'Podium Accuracy', 'Tiercé Exact', 'Tiercé Désordre', 'Quarté Exact']

        # Prepare data for radar chart
        radar_data = []
        for blend_value in blend_values:
            blend_str = str(blend_value)
            if blend_str in self.results:
                summary = self.results[blend_str].get('summary', {})
                radar_data.append([
                    summary.get('winner_accuracy', 0),
                    summary.get('podium_accuracy', 0),
                    summary.get('pmu_bets', {}).get('tierce_exact_rate', 0),
                    summary.get('pmu_bets', {}).get('tierce_desordre_rate', 0),
                    summary.get('pmu_bets', {}).get('quarte_exact_rate', 0)
                ])

        if radar_data:
            try:
                # Create radar chart
                self._create_radar_chart(
                    radar_data,
                    metrics_names,
                    [f"RF Weight {v}" for v in blend_values],
                    visualizations_dir / "radar_chart.png"
                )
            except Exception as e:
                print(f"Error creating radar chart: {str(e)}")
                self.logger.error(f"Error creating radar chart: {str(e)}")

        print(f"Visualizations saved to {visualizations_dir}")
        self.logger.info(f"Visualizations saved to {visualizations_dir}")

    def _create_radar_chart(self, data, labels, legend, output_file):
        """
        Create a radar chart for the metrics.

        Args:
            data: List of data series to plot
            labels: Labels for each axis
            legend: Legend entries for each data series
            output_file: Output file path
        """
        # Number of metrics
        N = len(labels)

        # Create angles for each metric
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

        # Close the polygon
        angles += angles[:1]

        # Create figure
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)

        # Plot each data series
        for i, series in enumerate(data):
            # Close the polygon
            values = series + [series[0]]

            # Plot values
            ax.plot(angles, values, 'o-', linewidth=2, label=legend[i])
            ax.fill(angles, values, alpha=0.1)

        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.title("Metric Comparison Across RF Weights", size=15)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    @classmethod
    def load_results(cls, results_dir: str) -> 'BlendingComparison':
        """
        Load results from a previous comparison run.

        Args:
            results_dir: Directory with saved results

        Returns:
            BlendingComparison instance with loaded results
        """
        results_path = Path(results_dir)
        results_file = results_path / "results.json"

        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

            # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)

            # Create instance
        comparison = cls(
            model_path="dummy_path",  # Will be overridden
            output_dir=str(results_path),
            blending_values=[float(k) for k in results.keys()]
        )

        # Set results
        comparison.results = results

        # Generate comparison and visualizations
        comparison._generate_comparison()
        comparison._visualize_results()

        return comparison

def main():
    parser = argparse.ArgumentParser(description="Compare race prediction results with different blending values")
    parser.add_argument("--model", type=str, required=True, help="Path to the model or model name")
    parser.add_argument("--db", type=str, help="Database name from config (defaults to active_db)")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--blend-values", type=str, help="Comma-separated list of blending values to test")
    parser.add_argument("--start-date", type=str, help="Start date for evaluation (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=3, help="Number of days to evaluate")
    parser.add_argument("--load", type=str, help="Load and analyze results from specified directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    if args.load:
        # Load previous results
        try:
            comparison = BlendingComparison.load_results(args.load)
            print(f"Loaded and analyzed results from {args.load}")
            return 0
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            return 1
    # Parse blend values
    if args.blend_values:
        blend_values = [float(v) for v in args.blend_values.split(",")]
    else:
        blend_values = [0.0, 0.25, 0.5, 0.7, 0.85, 1.0]  # Default values
    # Create dates list
    if args.start_date:
        start_date = args.start_date
        # Parse date to check format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date format: {start_date}. Use YYYY-MM-DD format.")
            return 1
        # Generate dates
        dates = []
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        for i in range(args.days):
            date_str = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(date_str)
    else:
        dates = None
    # Create comparison object
    comparison = BlendingComparison(
        model_path=args.model,
        db_name=args.db,
        output_dir=args.output,
        blending_values=blend_values,
        verbose=args.verbose
    )
    # Run evaluation
    if dates:
        result = comparison.evaluate_blending_values(dates=dates)
    else:
        result = comparison.evaluate_blending_values(num_days=args.days)
    print("\nBlending value comparison completed!")
    print(f"Results saved to: {comparison.output_dir}")
    print("\nRecommended blending value: " +
          f"{result['recommendation']['blend_value']:.2f} (score: {result['recommendation']['score']:.4f})")
    # Print best values for each metric
    best = result.get('best', {})
    print("\nBest values by metric:")
    metrics = [
        ("Winner Accuracy", best.get('winner', {})),
        ("Podium Accuracy", best.get('podium', {})),
        ("Tiercé Exact", best.get('tierce_exact', {})),
        ("Tiercé Désordre", best.get('tierce_desordre', {})),
        ("Quarté Exact", best.get('quarte_exact', {}))
    ]
    for metric_name, metric_data in metrics:
        value = metric_data.get('value', 0)
        blend = metric_data.get('blend')
        print(f"  - {metric_name}: {value:.4f} (blend={blend})")
    return 0
if __name__ == "__main__":
    sys.exit(main())