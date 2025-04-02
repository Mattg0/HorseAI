# race_prediction/predict_daily_races.py

import argparse
import datetime
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the orchestrator
from core.orchestrators.prediction_orchestrator import PredictionOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Predict daily horse races")
    parser.add_argument("--date", type=str, help="Date to process (YYYY-MM-DD format, default: today)")
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--db", type=str, help="Database name from config (defaults to active_db)")
    parser.add_argument("--race", type=str, help="Process a specific race by ID (comp)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate predictions against actual results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", type=str, help="Output file for JSON results")

    args = parser.parse_args()

    # Create the orchestrator
    orchestrator = PredictionOrchestrator(
        model_path=args.model,
        db_name=args.db,
        verbose=args.verbose
    )

    results = None

    # Process specific race if requested
    if args.race:
        if args.evaluate:
            # Evaluate a specific race
            results = orchestrator.evaluate_predictions(args.race)
            print(f"Evaluation for race {args.race}: {results['status']}")
            if results['status'] == 'success':
                metrics = results['metrics']
                print(f"  Winner correct: {metrics['winner_correct']}")
                print(f"  Podium accuracy: {metrics['podium_accuracy']:.2f}")
                print(f"  Mean rank error: {metrics['mean_rank_error']:.2f}")
                print(f"  Exacta correct: {metrics['exacta_correct']}")
                print(f"  Trifecta correct: {metrics['trifecta_correct']}")
            else:
                print(f"  Error: {results.get('error', 'Unknown error')}")
        else:
            # Predict a specific race
            results = orchestrator.predict_race(args.race)
            print(f"Prediction for race {args.race}: {results['status']}")
            if results['status'] == 'success':
                predictions = results['predictions']
                print(f"  Predicted top 3:")
                for i, horse in enumerate(predictions[:3]):
                    print(f"    {i + 1}. {horse['numero']} - {horse['cheval']} "
                          f"(predicted position: {horse['predicted_position']:.2f})")
            else:
                print(f"  Error: {results.get('error', 'Unknown error')}")
    else:
        # Process all races for a date
        date = args.date or datetime.datetime.now().strftime("%Y-%m-%d")

        if args.evaluate:
            # Evaluating requires fetching all races and evaluating them one by one
            races = orchestrator.race_fetcher.get_races_by_date(date)

            if not races:
                print(f"No races found for {date}")
                return

            print(f"Evaluating {len(races)} races for {date}")

            evaluation_results = []
            for race in races:
                comp = race['comp']
                result = orchestrator.evaluate_predictions(comp)
                evaluation_results.append(result)

            # Calculate summary metrics
            success_count = sum(1 for r in evaluation_results if r['status'] == 'success')
            winner_correct = sum(1 for r in evaluation_results
                                 if r['status'] == 'success' and r['metrics']['winner_correct'])

            if success_count > 0:
                podium_accuracy = sum(r['metrics']['podium_accuracy'] for r in evaluation_results
                                      if r['status'] == 'success') / success_count

                print(f"Evaluation summary for {date}:")
                print(f"  Races evaluated: {success_count}")
                print(f"  Winner accuracy: {winner_correct / success_count:.2f} ({winner_correct}/{success_count})")
                print(f"  Average podium accuracy: {podium_accuracy:.2f}")
            else:
                print(f"No races successfully evaluated for {date}")

            results = {
                'date': date,
                'evaluated': success_count,
                'winner_accuracy': winner_correct / success_count if success_count > 0 else 0,
                'evaluations': evaluation_results
            }
        else:
            # Predict all races for the date
            results = orchestrator.predict_races_by_date(date)

            print(f"Prediction results for {date}:")
            print(f"  Total races: {results['total_races']}")
            print(f"  Successfully predicted: {results['predicted']}")
            print(f"  Errors: {results['errors']}")
            print(f"  Skipped: {results['skipped']}")

    # Save results to file if requested
    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()