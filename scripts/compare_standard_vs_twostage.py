#!/usr/bin/env python3
"""
Compare standard predictions vs two-stage predictions against actual results.

This script:
1. Loads quinté races with actual results
2. Generates predictions with two-stage refinement
3. Compares BEFORE (standard) vs AFTER (two-stage) vs ACTUAL
4. Shows which approach is more accurate for positions 1-5

Usage:
    python3 scripts/compare_standard_vs_twostage.py [--date YYYY-MM-DD]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime


def get_actual_results_for_race(race_comp, db_path):
    """Get actual race results from database."""
    import sqlite3

    conn = sqlite3.connect(db_path)

    # Get results from daily_race table (column is 'actual_results')
    query = """
    SELECT actual_results FROM daily_race
    WHERE comp = ?
    """

    cursor = conn.execute(query, (race_comp,))
    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        # Parse actual_results string (format: "1-5-3-2-4")
        actual_results = result[0]
        if actual_results and '-' in str(actual_results):
            # Handle both string and potential JSON format
            if isinstance(actual_results, str) and '-' in actual_results:
                actual_order = [int(x) for x in actual_results.split('-')]
                return actual_order

    return None


def calculate_position_accuracy(predicted_order, actual_order):
    """
    Calculate accuracy metrics for predicted vs actual order.

    Returns dict with:
    - winner_correct: bool
    - top3_accuracy: float (0-1)
    - top5_accuracy: float (0-1)
    - position_4_correct: bool
    - position_5_correct: bool
    - mean_rank_error: float
    """
    if not actual_order or len(actual_order) < 5:
        return None

    actual_top5 = actual_order[:5]
    predicted_top5 = predicted_order[:5]

    # Winner correct
    winner_correct = predicted_top5[0] == actual_top5[0]

    # Top 3 accuracy
    predicted_top3_set = set(predicted_top5[:3])
    actual_top3_set = set(actual_top5[:3])
    top3_accuracy = len(predicted_top3_set & actual_top3_set) / 3.0

    # Top 5 accuracy
    predicted_top5_set = set(predicted_top5)
    actual_top5_set = set(actual_top5)
    top5_accuracy = len(predicted_top5_set & actual_top5_set) / 5.0

    # Position 4 and 5 correct (exact position)
    position_4_correct = (len(predicted_top5) >= 4 and len(actual_top5) >= 4 and
                          predicted_top5[3] == actual_top5[3])
    position_5_correct = (len(predicted_top5) >= 5 and len(actual_top5) >= 5 and
                          predicted_top5[4] == actual_top5[4])

    # Position 4 and 5 in top 5 (any order)
    position_4_in_top5 = (len(actual_top5) >= 4 and actual_top5[3] in predicted_top5)
    position_5_in_top5 = (len(actual_top5) >= 5 and actual_top5[4] in predicted_top5)

    # Mean rank error for top 5
    rank_errors = []
    for i, horse in enumerate(predicted_top5, 1):
        if horse in actual_top5:
            actual_rank = actual_top5.index(horse) + 1
            rank_errors.append(abs(i - actual_rank))

    mean_rank_error = np.mean(rank_errors) if rank_errors else float('inf')

    return {
        'winner_correct': winner_correct,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'position_4_correct': position_4_correct,
        'position_5_correct': position_5_correct,
        'position_4_in_top5': position_4_in_top5,
        'position_5_in_top5': position_5_in_top5,
        'mean_rank_error': mean_rank_error
    }


def compare_predictions(race_date=None):
    """
    Run comparison between standard and two-stage predictions.
    """
    print("\n" + "="*80)
    print("COMPARING STANDARD vs TWO-STAGE PREDICTIONS")
    print("="*80 + "\n")

    from race_prediction.predict_quinte import QuintePredictionEngine
    from utils.env_setup import get_sqlite_dbpath

    # Get database path
    db_path = get_sqlite_dbpath('2years')

    # Create predictor
    print("[1/4] Creating QuintePredictionEngine...")
    predictor = QuintePredictionEngine(verbose=False)  # Disable verbose for cleaner output
    print("✓ Predictor created\n")

    # Load races
    print("[2/4] Loading quinté races...")
    df_races = predictor.load_daily_quinte_races(race_date=race_date)

    if len(df_races) == 0:
        print("✗ No quinté races found")
        return

    print(f"✓ Loaded {len(df_races)} quinté races\n")

    # Filter to races with actual results
    races_with_results = []
    for _, race in df_races.iterrows():
        actual_order = get_actual_results_for_race(race['comp'], db_path)
        if actual_order and len(actual_order) >= 5:
            races_with_results.append((race, actual_order))

    if not races_with_results:
        print("⚠ No races with actual results found")
        print("  This comparison requires races that have already finished")
        print("\nFor testing purposes, showing predictions without comparison...")

        # Just show one race prediction
        test_race = df_races.iloc[0]
        result = predictor.run_prediction(
            race_date=test_race['jour'],
            output_dir='predictions/comparison',
            store_to_db=False
        )

        if result['status'] == 'success':
            df_pred = result['predictions']
            race_comp = df_pred['comp'].iloc[0]
            race_df = df_pred[df_pred['comp'] == race_comp].sort_values('predicted_position')

            print(f"\nRace: {race_comp}")
            print("\nPredictions (with two-stage refinement):")
            print(f"{'Rank':<6} {'#':<4} {'Horse':<25} {'Odds':<8} {'Pred Pos':<10} {'Notes'}")
            print("-" * 80)

            top5 = race_df.head(5)
            for idx, (_, row) in enumerate(top5.iterrows(), 1):
                horse_name = row.get('nom', 'Unknown')[:24]
                odds = row.get('cotedirect', 0.0)
                pos = row['predicted_position']

                # Check if this was adjusted
                notes = ""
                if 'original_predicted_position' in row.index and pd.notna(row['original_predicted_position']):
                    orig_pos = row['original_predicted_position']
                    if abs(orig_pos - pos) > 0.1:
                        notes = f"(was {orig_pos:.2f})"
                        if 15 <= odds <= 35:
                            notes += " ⭐LONGSHOT"

                print(f"{idx:<6} {int(row['numero']):<4} {horse_name:<25} {odds:<8.1f} {pos:<10.2f} {notes}")

        return

    print(f"✓ Found {len(races_with_results)} races with actual results\n")

    # Run predictions with two-stage enabled (current state)
    print("[3/4] Running predictions with two-stage refinement...")

    all_comparisons = []

    for race, actual_order in races_with_results:
        race_comp = race['comp']
        race_date = race['jour']

        print(f"\n  Processing race {race_comp}...")

        # Run prediction
        result = predictor.run_prediction(
            race_date=race_date,
            output_dir='predictions/comparison',
            store_to_db=False
        )

        if result['status'] != 'success':
            print(f"    ✗ Prediction failed")
            continue

        df_pred = result['predictions']
        race_df = df_pred[df_pred['comp'] == race_comp].copy()

        # Get BEFORE two-stage (original predictions)
        if 'original_predicted_position' in race_df.columns:
            race_df_before = race_df.copy()
            race_df_before['predicted_position'] = race_df_before['original_predicted_position']
            race_df_before = race_df_before.sort_values('predicted_position')
            before_order = race_df_before.head(5)['numero'].tolist()
        else:
            print("    ⚠ original_predicted_position not found, skipping comparison")
            continue

        # Get AFTER two-stage (current predictions)
        race_df_after = race_df.sort_values('predicted_position')
        after_order = race_df_after.head(5)['numero'].tolist()

        # Calculate accuracies
        before_metrics = calculate_position_accuracy(before_order, actual_order)
        after_metrics = calculate_position_accuracy(after_order, actual_order)

        comparison = {
            'race_comp': race_comp,
            'actual_order': actual_order[:5],
            'before_order': before_order,
            'after_order': after_order,
            'before_metrics': before_metrics,
            'after_metrics': after_metrics,
            'race_df': race_df
        }

        all_comparisons.append(comparison)
        print(f"    ✓ Analyzed")

    if not all_comparisons:
        print("\n✗ No valid comparisons generated")
        return

    print(f"\n[4/4] Comparing results...\n")
    print("="*80)

    # Aggregate statistics
    total_races = len(all_comparisons)

    # Count improvements
    winner_same = 0
    winner_better = 0
    winner_worse = 0

    pos4_before_correct = 0
    pos4_after_correct = 0
    pos4_before_in_top5 = 0
    pos4_after_in_top5 = 0

    pos5_before_correct = 0
    pos5_after_correct = 0
    pos5_before_in_top5 = 0
    pos5_after_in_top5 = 0

    top5_before_acc = []
    top5_after_acc = []

    for comp in all_comparisons:
        # Winner accuracy
        if comp['before_metrics']['winner_correct'] == comp['after_metrics']['winner_correct']:
            winner_same += 1
        elif comp['after_metrics']['winner_correct']:
            winner_better += 1
        elif comp['before_metrics']['winner_correct']:
            winner_worse += 1

        # Position 4-5 accuracy
        if comp['before_metrics']['position_4_correct']:
            pos4_before_correct += 1
        if comp['after_metrics']['position_4_correct']:
            pos4_after_correct += 1
        if comp['before_metrics']['position_4_in_top5']:
            pos4_before_in_top5 += 1
        if comp['after_metrics']['position_4_in_top5']:
            pos4_after_in_top5 += 1

        if comp['before_metrics']['position_5_correct']:
            pos5_before_correct += 1
        if comp['after_metrics']['position_5_correct']:
            pos5_after_correct += 1
        if comp['before_metrics']['position_5_in_top5']:
            pos5_before_in_top5 += 1
        if comp['after_metrics']['position_5_in_top5']:
            pos5_after_in_top5 += 1

        # Top 5 accuracy
        top5_before_acc.append(comp['before_metrics']['top5_accuracy'])
        top5_after_acc.append(comp['after_metrics']['top5_accuracy'])

    # Print summary
    print(f"SUMMARY ({total_races} races analyzed)")
    print("="*80)

    print("\n1. WINNER PREDICTION:")
    print(f"   Standard:     {sum(c['before_metrics']['winner_correct'] for c in all_comparisons)}/{total_races} ({sum(c['before_metrics']['winner_correct'] for c in all_comparisons)/total_races*100:.1f}%)")
    print(f"   Two-stage:    {sum(c['after_metrics']['winner_correct'] for c in all_comparisons)}/{total_races} ({sum(c['after_metrics']['winner_correct'] for c in all_comparisons)/total_races*100:.1f}%)")
    if winner_better > winner_worse:
        print(f"   → Two-stage BETTER by {winner_better - winner_worse} races")
    elif winner_worse > winner_better:
        print(f"   → Standard BETTER by {winner_worse - winner_better} races")
    else:
        print(f"   → Same performance")

    print("\n2. POSITION 4 ACCURACY (exact position):")
    print(f"   Standard:     {pos4_before_correct}/{total_races} ({pos4_before_correct/total_races*100:.1f}%)")
    print(f"   Two-stage:    {pos4_after_correct}/{total_races} ({pos4_after_correct/total_races*100:.1f}%)")
    diff = pos4_after_correct - pos4_before_correct
    if diff > 0:
        print(f"   → Two-stage BETTER by {diff} races (+{diff/total_races*100:.1f}%)")
    elif diff < 0:
        print(f"   → Standard BETTER by {abs(diff)} races ({diff/total_races*100:.1f}%)")
    else:
        print(f"   → Same performance")

    print("\n3. POSITION 5 ACCURACY (exact position):")
    print(f"   Standard:     {pos5_before_correct}/{total_races} ({pos5_before_correct/total_races*100:.1f}%)")
    print(f"   Two-stage:    {pos5_after_correct}/{total_races} ({pos5_after_correct/total_races*100:.1f}%)")
    diff = pos5_after_correct - pos5_before_correct
    if diff > 0:
        print(f"   → Two-stage BETTER by {diff} races (+{diff/total_races*100:.1f}%)")
    elif diff < 0:
        print(f"   → Standard BETTER by {abs(diff)} races ({diff/total_races*100:.1f}%)")
    else:
        print(f"   → Same performance")

    print("\n4. POSITION 4 IN TOP 5 (any order):")
    print(f"   Standard:     {pos4_before_in_top5}/{total_races} ({pos4_before_in_top5/total_races*100:.1f}%)")
    print(f"   Two-stage:    {pos4_after_in_top5}/{total_races} ({pos4_after_in_top5/total_races*100:.1f}%)")

    print("\n5. POSITION 5 IN TOP 5 (any order):")
    print(f"   Standard:     {pos5_before_in_top5}/{total_races} ({pos5_before_in_top5/total_races*100:.1f}%)")
    print(f"   Two-stage:    {pos5_after_in_top5}/{total_races} ({pos5_after_in_top5/total_races*100:.1f}%)")

    print("\n6. TOP 5 ACCURACY (all 5 horses in any order):")
    print(f"   Standard:     {np.mean(top5_before_acc)*100:.1f}%")
    print(f"   Two-stage:    {np.mean(top5_after_acc)*100:.1f}%")
    diff = (np.mean(top5_after_acc) - np.mean(top5_before_acc)) * 100
    if diff > 0:
        print(f"   → Two-stage BETTER by {diff:.1f}%")
    elif diff < 0:
        print(f"   → Standard BETTER by {abs(diff):.1f}%")

    # Detailed race-by-race comparison
    print("\n" + "="*80)
    print("DETAILED RACE-BY-RACE COMPARISON")
    print("="*80)

    for comp in all_comparisons:
        print(f"\nRace: {comp['race_comp']}")
        print(f"Actual:     {'-'.join(str(x) for x in comp['actual_order'])}")
        print(f"Standard:   {'-'.join(str(x) for x in comp['before_order'])}")
        print(f"Two-stage:  {'-'.join(str(x) for x in comp['after_order'])}")

        # Show which changed
        changes = []
        for i in range(5):
            if comp['before_order'][i] != comp['after_order'][i]:
                before_horse = comp['before_order'][i]
                after_horse = comp['after_order'][i]

                # Get odds
                race_df = comp['race_df']
                before_odds = race_df[race_df['numero'] == before_horse]['cotedirect'].values[0] if before_horse in race_df['numero'].values else 0
                after_odds = race_df[race_df['numero'] == after_horse]['cotedirect'].values[0] if after_horse in race_df['numero'].values else 0

                changes.append(f"pos {i+1}: #{before_horse} (odds {before_odds:.1f}) → #{after_horse} (odds {after_odds:.1f})")

        if changes:
            print(f"Changes:    {', '.join(changes)}")
        else:
            print(f"Changes:    None")

        # Show accuracy
        print(f"Pos 4:      Standard={comp['before_metrics']['position_4_correct']}, Two-stage={comp['after_metrics']['position_4_correct']}")
        print(f"Pos 5:      Standard={comp['before_metrics']['position_5_correct']}, Two-stage={comp['after_metrics']['position_5_correct']}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    total_improvements = (pos4_after_correct - pos4_before_correct) + (pos5_after_correct - pos5_before_correct)

    if total_improvements > 0:
        print(f"\n✓ Two-stage prediction improves positions 4-5 accuracy by {total_improvements} correct positions across {total_races} races")
        print(f"  This is a {total_improvements/(total_races*2)*100:.1f}% improvement in positions 4-5")
    elif total_improvements < 0:
        print(f"\n✗ Two-stage prediction decreases positions 4-5 accuracy by {abs(total_improvements)} correct positions")
        print(f"  Consider disabling two-stage refinement")
    else:
        print(f"\n→ Two-stage prediction has the same accuracy as standard prediction")
        print(f"  May need more races to see statistically significant differences")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare standard vs two-stage quinté predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--date', type=str,
                       help='Specific date (YYYY-MM-DD) or None for all races with results')

    args = parser.parse_args()

    try:
        compare_predictions(race_date=args.date)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
