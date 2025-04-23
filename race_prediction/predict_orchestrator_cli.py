#!/usr/bin/env python
# race_prediction/predict_orchestrator_cli.py

import argparse
import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the orchestrator
from core.orchestrators.prediction_orchestrator import PredictionOrchestrator


def report_evaluation_results(results):
    """
    Generate a formatted report for race evaluation results.

    Args:
        results: Evaluation results dictionary

    Returns:
        Formatted string with evaluation report
    """
    if not results or 'status' not in results:
        return "Invalid results format"

    if results['status'] != 'success':
        return f"Error: {results.get('error', 'Unknown error')}"

    # Extract metrics
    metrics = results.get('metrics', {})
    race_info = metrics.get('race_info', {})

    # Format race information
    race_header = (
        f"\nEvaluation for race {race_info.get('comp')}: "
        f"{race_info.get('hippo')} - {race_info.get('prix')} "
        f"({race_info.get('jour')})"
    )

    # Format basic metrics
    basic_metrics = (
        f"Basic metrics:\n"
        f"  Winner correctly predicted: {'✓' if metrics.get('winner_correct') else '✗'}\n"
        f"  Podium accuracy: {metrics.get('podium_accuracy', 0):.2f}\n"
        f"  Mean rank error: {metrics.get('mean_rank_error', 'N/A'):.2f}"
    )

    # Format PMU bet results
    pmu_bets = metrics.get('pmu_bets', {})
    winning_bets = metrics.get('winning_bets', [])

    if not winning_bets:
        bet_results = "PMU bet results: No winning bets"
    else:
        bet_results = "PMU bet results: ✓ " + ", ".join([
            format_bet_name(bet_type) for bet_type in winning_bets
        ])

    # Format arrival orders
    arrival_info = (
        f"Arrival orders:\n"
        f"  Predicted: {metrics.get('predicted_arriv', 'N/A')}\n"
        f"  Actual: {metrics.get('actual_arriv', 'N/A')}"
    )

    # Combine all sections
    report = f"{race_header}\n\n{basic_metrics}\n\n{bet_results}\n\n{arrival_info}"
    return report


def format_bet_name(bet_type):
    """Format bet type names for display"""
    name_mapping = {
        'tierce_exact': 'Tiercé Exact',
        'tierce_desordre': 'Tiercé Désordre',
        'quarte_exact': 'Quarté Exact',
        'quarte_desordre': 'Quarté Désordre',
        'quinte_exact': 'Quinté+ Exact',
        'quinte_desordre': 'Quinté+ Désordre',
        'bonus4': 'Bonus 4',
        'bonus3': 'Bonus 3',
        'deuxsur4': '2 sur 4',
        'multi4': 'Multi en 4'
    }
    return name_mapping.get(bet_type, bet_type)


def report_summary_evaluation(summary):
    """
    Generate a formatted report for evaluation summary.

    Args:
        summary: Evaluation summary dictionary

    Returns:
        Formatted string with summary report
    """
    if not summary or 'summary_metrics' not in summary:
        return "No summary metrics available"

    metrics = summary['summary_metrics']
    races_evaluated = metrics.get('races_evaluated', 0)

    if races_evaluated == 0:
        return "No races evaluated"

    # Basic statistics
    basic_stats = (
        f"\nEvaluation Summary ({races_evaluated} races):\n"
        f"  Winner accuracy: {metrics.get('winner_accuracy', 0):.2f} "
        f"({int(metrics.get('winner_accuracy', 0) * races_evaluated)}/{races_evaluated})\n"
        f"  Average podium accuracy: {metrics.get('avg_podium_accuracy', 0):.2f}\n"
        f"  Average mean rank error: {metrics.get('avg_mean_rank_error', 'N/A'):.2f}"
    )

    # PMU bet statistics
    bet_stats = metrics.get('bet_statistics', {})
    pmu_summary = (
        f"\nOverall PMU Bet Performance:\n"
        f"  Races with at least one winning bet: {bet_stats.get('races_with_wins', 0)} "
        f"({bet_stats.get('win_rate', 0):.2f})\n"
        f"  Races with no winning bets: {bet_stats.get('races_with_no_wins', 0)}"
    )

    # Distribution of races by bet count
    bet_counts = bet_stats.get('races_by_bet_count', {})
    if bet_counts:
        bet_count_lines = [
            f"  Races winning {count} bet types: {bet_counts.get(count, 0)}"
            for count in sorted(bet_counts.keys())
        ]
        bet_count_summary = "\n" + "\n".join(bet_count_lines)
    else:
        bet_count_summary = ""

    # Detailed bet type performance
    bet_type_summary = metrics.get('bet_type_summary', {})
    if bet_type_summary:
        bet_type_lines = []
        for bet_type, stats in bet_type_summary.items():
            bet_type_lines.append(
                f"  {format_bet_name(bet_type)}: "
                f"{stats.get('wins', 0)}/{stats.get('total_races', 0)} "
                f"({stats.get('success_rate', 0):.1f}%)"
            )
        bet_type_report = "\n\nPMU Bet Type Success Rates:\n" + "\n".join(bet_type_lines)
    else:
        bet_type_report = ""

    # Complete report
    report = f"{basic_stats}\n{pmu_summary}{bet_count_summary}{bet_type_report}"
    return report


def main():
    """Command-line interface for race prediction orchestration."""
    parser = argparse.ArgumentParser(description="Orchestrate race predictions workflow")
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--db", type=str, help="Database name from config (defaults to active_db)")
    parser.add_argument("--date", type=str, help="Date to process (YYYY-MM-DD format, default: today)")
    parser.add_argument("--race", type=str, help="Process a specific race by ID (comp)")
    parser.add_argument("--action", type=str, choices=['predict', 'evaluate', 'fetch', 'fetchpredict'],
                        default='predict', help="Action to perform")
    parser.add_argument("--blend", type=float, default=0.7, help="Blend weight for RF model (0-1)")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--summary", action="store_true", help="Show only summary info, not detailed results")

    args = parser.parse_args()

    # Create the orchestrator
    orchestrator = PredictionOrchestrator(
        model_path=args.model,
        db_name=args.db,
        verbose=args.verbose
    )

    results = None

    # Process specific race if provided
    if args.race:
        comp = args.race

        if args.action == 'predict':
            if not args.verbose:
                print(f"Generating predictions for race {comp}...")
            results = orchestrator.predict_race(comp, blend_weight=args.blend)

            if results['status'] == 'success':
                print(f"\nPredictions for race {comp}:")
                print(f"  Race: {results['metadata']['hippo']} - {results['metadata']['prix']}")
                print(f"\nPredicted order of finish:")

                for i, horse in enumerate(results['predictions'][:5]):
                    print(f"  {i + 1}. {horse['numero']} - {horse['cheval']} "
                          f"(predicted: {horse['predicted_position']:.2f})")

                if len(results['predictions']) > 5:
                    print(f"  ... and {len(results['predictions']) - 5} more horses")
            else:
                print(f"Error: {results.get('error', 'Unknown error')}")

        elif args.action == 'evaluate':
            results = orchestrator.evaluate_predictions(comp)

            if results['status'] == 'success':
                metrics = results['metrics']
                bet_results = metrics.get('winning_bets', [])

                print(f"####Evaluation for race {comp}:#######")
                print(f"  Race: {metrics['race_info'].get('hippo')} - {metrics['race_info'].get('prix')}")
                print(f"Arrivals:")
                print(f"  Predicted: {metrics['predicted_arriv']}")
                print(f"  Actual:    {metrics['actual_arriv']}")
                print(f"  Winner prediction: {'✓' if metrics['winner_correct'] else '✗'}")
                print(f"  Podium accuracy: {metrics['podium_accuracy']:.2f}")
                print(f"Bets:")
                if bet_results:
                    print("  Winning bets:")
                    for bet in bet_results:
                        print(f"  ✓ {format_bet_name(bet)}")
                else:
                    print("  No winning bets")
            else:
                print(f"Error: {results.get('error', 'Unknown error')}")

    # Process all races for a date
    else:
        # Use today's date if none provided
        date = args.date or datetime.now().strftime("%Y-%m-%d")

        if args.action == 'predict':
            if not args.verbose:
                print(f"Generating predictions for races on {date}...")
            results = orchestrator.predict_races_by_date(date, blend_weight=args.blend)

            print(f"\nPrediction summary for {date}:")
            print(f"  Total races: {results['total_races']}")
            print(f"  Successfully predicted: {results['predicted']}")
            print(f"  Errors: {results['errors']}")
            print(f"  Skipped: {results['skipped']}")

            if not args.summary and results.get('results'):
                print("\nPredicted races:")
                for race in results['results']:
                    if race['status'] == 'success':
                        print(f"  ✓ {race['comp']}: {race.get('metadata', {}).get('hippo', 'Unknown')}")
                    else:
                        print(f"  ✗ {race['comp']}: {race.get('error', 'Failed')}")

        elif args.action == 'evaluate':
            if not args.verbose:
                print(f"Evaluating predictions for races on {date}...")
            results = orchestrator.evaluate_predictions_by_date(date)

            if results.get('summary_metrics'):
                metrics = results['summary_metrics']
                races_evaluated = metrics.get('races_evaluated', 0)
                bet_stats = metrics.get('bet_statistics', {})

                print(f"\nEvaluation Summary ({races_evaluated} races):")
                print(
                    f"  Winner accuracy: {metrics.get('winner_accuracy', 0):.2f} ({int(metrics.get('winner_accuracy', 0) * races_evaluated)}/{races_evaluated})")
                print(
                    f"  Races with winning bets: {bet_stats.get('races_with_wins', 0)}/{races_evaluated} ({bet_stats.get('win_rate', 0) * 100:.1f}%)")

                # Show top 3 most successful bet types
                bet_summary = metrics.get('bet_type_summary', {})
                if bet_summary:
                    sorted_bets = sorted(bet_summary.items(), key=lambda x: x[1]['success_rate'], reverse=True)[:3]
                    print("\nTop performing bet types:")
                    for bet_type, stats in sorted_bets:
                        print(
                            f"  {format_bet_name(bet_type)}: {stats['wins']}/{stats['total_races']} ({stats['success_rate']:.1f}%)")

                if not args.summary and results.get('results'):
                    print("\nResults by race:")
                    for race in [r for r in results['results'] if r['status'] == 'success']:
                        metrics = race['metrics']
                        winning_bets = metrics.get('winning_bets', [])
                        bet_count = len(winning_bets)

                        print(f"  {race['comp']}: Winning Bets {'✓' if bet_count> 0 else '0'} ")
            else:
                print("No metrics available. Races may not have been evaluated yet.")

        elif args.action == 'fetch':
            if not args.verbose:
                print(f"Fetching races for {date}...")
            results = orchestrator.race_fetcher.fetch_and_store_daily_races(date)

            print(f"\nFetch summary for {date}:")
            print(f"  Total races: {results.get('total_races', 0)}")
            print(f"  Successfully processed: {results.get('successful', 0)}")
            print(f"  Failed: {results.get('failed', 0)}")

        elif args.action == 'fetchpredict':
            if not args.verbose:
                print(f"Fetching and predicting races for {date}...")
            results = orchestrator.fetch_and_predict_races(date, blend_weight=args.blend)

            fetch_results = results.get('fetch_results', {})
            prediction_results = results.get('prediction_results', {})

            print(f"\nFetch and predict for {date}:")
            print(f"  Fetched: {fetch_results.get('total_races', 0)} races")
            print(
                f"  Predicted: {prediction_results.get('predicted', 0)}/{prediction_results.get('total_races', 0)} races")

    # Save results to file if requested
    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_path}")

    return 0


# Helper function for bet type formatting
def format_bet_name(bet_type):
    """Format bet type names for display"""
    name_mapping = {
        'tierce_exact': 'Tiercé Exact',
        'tierce_desordre': 'Tiercé Désordre',
        'quarte_exact': 'Quarté Exact',
        'quarte_desordre': 'Quarté Désordre',
        'quinte_exact': 'Quinté+ Exact',
        'quinte_desordre': 'Quinté+ Désordre',
        'bonus4': 'Bonus 4',
        'bonus3': 'Bonus 3',
        'deuxsur4': '2 sur 4',
        'multi4': 'Multi en 4'
    }
    return name_mapping.get(bet_type, bet_type)


#if __name__ == "__main__":
#    sys.exit(main())


def debug_fetchpredict(date, model_path, db_name=None, blend_weight=0.7, verbose=False):
    """
    Debug function to execute fetchpredict for a specific date and model.
    This can be called directly from the IDE for setting breakpoints.

    Args:
        date: Date string in format YYYY-MM-DD
        model_path: Path to the model directory
        db_name: Database name from config (defaults to active_db)
        blend_weight: Weight for RF model in blend (0-1)
        verbose: Whether to enable verbose output

    Returns:
        Results dictionary from fetch_and_predict_races
    """
    from core.orchestrators.prediction_orchestrator import PredictionOrchestrator

    print(f"Debug: Starting fetchpredict for date {date} with model {model_path}")

    # Create orchestrator instance
    orchestrator = PredictionOrchestrator(
        model_path=model_path,
        db_name=db_name,
        verbose=verbose
    )

    # Execute fetch and predict
    results= orchestrator.fetch_and_predict_races(date,"0.7")
    #results = orchestrator.predict_races_by_date(date,"0.7")

    # Print summary results
    fetch_results = results.get('fetch_results', {})
    prediction_results = results.get('prediction_results', {})

    print(f"\nFetch summary for {date}:")
    print(f"  Total races: {fetch_results.get('total_races', 0)}")
    print(f"  Successfully processed: {fetch_results.get('successful', 0)}")
    print(f"  Failed: {fetch_results.get('failed', 0)}")

    print(f"\nPrediction summary for {date}:")
    print(f"  Total races: {prediction_results.get('total_races', 0)}")
    print(f"  Successfully predicted: {prediction_results.get('predicted', 0)}")
    print(f"  Errors: {prediction_results.get('errors', 0)}")
    print(f"  Skipped: {prediction_results.get('skipped', 0)}")

    return results


# This can be used at the end of the file for direct execution in IDE
if __name__ == "__main__":
    # For debug via IDE - uncomment the line below to use
     debug_fetchpredict( "2025-04-04", "models/2years/hybrid/2years_full_v20250409", verbose=True)

    # For normal CLI execution
    #sys.exit(main())