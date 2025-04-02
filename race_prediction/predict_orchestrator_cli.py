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


def main():
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

    # Use today's date if none provided
    date = args.date or datetime.now().strftime("%Y-%m-%d")

    # Process specific race if provided
    if args.race:
        comp = args.race

        if args.action == 'predict':
            print(f"Generating predictions for race {comp}...")
            results = orchestrator.predict_race(comp, blend_weight=args.blend)

            if results['status'] == 'success':
                print(f"\nPredictions for race {comp}:")
                print(f"  Race: {results['metadata']['hippo']} - {results['metadata']['prix']}")
                print(f"  Type: {results['metadata']['typec']}")
                print(f"  Date: {results['metadata']['jour']}")
                print(f"\nPredicted order of finish:")

                for i, horse in enumerate(results['predictions'][:5]):
                    print(f"  {i + 1}. {horse['numero']} - {horse['cheval']} "
                          f"(predicted position: {horse['predicted_position']:.2f})")

                if len(results['predictions']) > 5:
                    print(f"  ... and {len(results['predictions']) - 5} more horses")
            else:
                print(f"Error: {results.get('error', 'Unknown error')}")

        elif args.action == 'evaluate':
            print(f"Evaluating predictions for race {comp}...")
            results = orchestrator.evaluate_predictions(comp)

            if results['status'] == 'success':
                metrics = results['metrics']
                print(f"\nEvaluation for race {comp}:")
                print(f"  Race: {metrics['race_info'].get('hippo')} - {metrics['race_info'].get('prix')}")
                print(f"  Date: {metrics['race_info'].get('jour')}")
                print(f"\nPerformance metrics:")
                print(f"  Winner correctly predicted: {'✓' if metrics['winner_correct'] else '✗'}")
                print(f"  Podium accuracy: {metrics['podium_accuracy']:.2f}")
                print(f"  Mean rank error: {metrics['mean_rank_error']:.2f}")
                print(f"  Exacta correct: {'✓' if metrics['exacta_correct'] else '✗'}")
                print(f"  Trifecta correct: {'✓' if metrics['trifecta_correct'] else '✗'}")

                if not args.summary:
                    print(f"\nDetailed results:")
                    for i, horse in enumerate(metrics['details'][:10]):
                        print(f"  {i + 1}. {horse['numero']} - {horse['cheval']}: "
                              f"Predicted #{horse['predicted_rank']}, "
                              f"Actual #{horse['actual_rank']} "
                              f"(Error: {horse['rank_error']})")
            else:
                print(f"Error: {results.get('error', 'Unknown error')}")

    # Process all races for a date
    else:
        if args.action == 'predict':
            print(f"Generating predictions for races on {date}...")
            results = orchestrator.predict_races_by_date(date, blend_weight=args.blend)

            print(f"\nPrediction summary for {date}:")
            print(f"  Total races: {results['total_races']}")
            print(f"  Successfully predicted: {results['predicted']}")
            print(f"  Errors: {results['errors']}")
            print(f"  Skipped: {results['skipped']}")

            if not args.summary and results.get('results'):
                print("\nRace details:")
                for race in results['results']:
                    status_icon = "✓" if race['status'] == 'success' else "✗" if race['status'] == 'error' else "⚠"
                    print(f"  {status_icon} {race['comp']}: {race['status']}")

        elif args.action == 'evaluate':
            print(f"Evaluating predictions for races on {date}...")
            results = orchestrator.evaluate_predictions_by_date(date)

            if results.get('summary_metrics'):
                metrics = results['summary_metrics']
                pmu = metrics['pmu_bets']
                print("\nPMU Bet Type Success Rates:")
                print(
                    f"  Tiercé Exact: {pmu.get('tierce_exact_rate', 0):.2f} ({pmu.get('tierce_exact', 0)}/{metrics['races_evaluated']})")
                print(
                    f"  Tiercé Désordre: {pmu.get('tierce_desordre_rate', 0):.2f} ({pmu.get('tierce_desordre', 0)}/{metrics['races_evaluated']})")
                print(
                    f"  Quarté Exact: {pmu.get('quarte_exact_rate', 0):.2f} ({pmu.get('quarte_exact', 0)}/{metrics['races_evaluated']})")
                print(
                    f"  Quarté Désordre: {pmu.get('quarte_desordre_rate', 0):.2f} ({pmu.get('quarte_desordre', 0)}/{metrics['races_evaluated']})")
                print(
                    f"  Quinté+ Exact: {pmu.get('quinte_exact_rate', 0):.2f} ({pmu.get('quinte_exact', 0)}/{metrics['races_evaluated']})")
                print(
                    f"  Quinté+ Désordre: {pmu.get('quinte_desordre_rate', 0):.2f} ({pmu.get('quinte_desordre', 0)}/{metrics['races_evaluated']})")
                print(
                    f"  Bonus 4: {pmu.get('bonus4_rate', 0):.2f} ({pmu.get('bonus4', 0)}/{metrics['races_evaluated']})")
                print(
                    f"  Bonus 3: {pmu.get('bonus3_rate', 0):.2f} ({pmu.get('bonus3', 0)}/{metrics['races_evaluated']})")
                print(
                    f"  2 sur 4: {pmu.get('deuxsur4_rate', 0):.2f} ({pmu.get('deuxsur4', 0)}/{metrics['races_evaluated']})")
                print(
                    f"  Multi en 4: {pmu.get('multi4_rate', 0):.2f} ({pmu.get('multi4', 0)}/{metrics['races_evaluated']})")

                if not args.summary and results.get('results'):
                    print("\nRace details:")
                    for race in [r for r in results['results'] if r['status'] == 'success']:
                        comp = race['comp']
                        m = race['metrics']
                        winner = "✓" if m['winner_correct'] else "✗"
                        print(f"  {comp}: Winner {winner}, Podium {m['podium_accuracy']:.2f}, "
                              f"Error {m['mean_rank_error']:.2f}")
            else:
                print("No metrics available. Races may not have been evaluated yet.")

        elif args.action == 'fetch':
            print(f"Fetching races for {date}...")
            results = orchestrator.race_fetcher.fetch_and_store_daily_races(date)

            if results.get('total_races'):
                print(f"\nFetch summary for {date}:")
                print(f"  Total races: {results['total_races']}")
                print(f"  Successfully processed: {results['successful']}")
                print(f"  Failed: {results['failed']}")

                if not args.summary and results.get('races'):
                    print("\nRace details:")
                    for race in results['races']:
                        status_icon = "✓" if race['status'] == 'success' else "✗"
                        print(
                            f"  {status_icon} {race['comp']}: {race.get('hippo', '')} - {race.get('partant', 0)} runners")
            else:
                print(f"Error: {results.get('error', 'Unknown error')}")

        elif args.action == 'fetchpredict':
            print(f"Fetching and predicting races for {date}...")
            results = orchestrator.fetch_and_predict_races(date, blend_weight=args.blend)

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

            if not args.summary:
                print("\nRace status details:")
                for race in prediction_results.get('results', []):
                    status_icon = "✓" if race['status'] == 'success' else "✗" if race['status'] == 'error' else "⚠"
                    print(f"  {status_icon} {race['comp']}: {race['status']}")

    # Save results to file if requested
    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
