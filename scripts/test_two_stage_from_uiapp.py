#!/usr/bin/env python3
"""
Simple test to run from UIApp Python console or as a standalone script.

This tests the two-stage prediction on REAL quinté races from the database.

Usage from UIApp Python console:
>>> exec(open('scripts/test_two_stage_from_uiapp.py').read())

Or as standalone:
>>> python3 scripts/test_two_stage_from_uiapp.py
"""

def test_two_stage_with_real_data():
    """Test two-stage prediction with real quinté data."""
    print("\n" + "="*80)
    print("TESTING TWO-STAGE PREDICTION WITH REAL DATABASE DATA")
    print("="*80 + "\n")

    from race_prediction.predict_quinte import QuintePredictionEngine
    import pandas as pd

    # Create predictor
    print("[1/5] Creating QuintePredictionEngine...")
    predictor = QuintePredictionEngine(verbose=True)
    print("✓ Predictor created\n")

    # Load quinté races
    print("[2/5] Loading quinté races from database...")
    df_races = predictor.load_daily_quinte_races(race_date=None)

    if len(df_races) == 0:
        print("✗ No quinté races found in database")
        print("\nTo add races, use UIApp:")
        print("  1. Go to 'Data Management' tab")
        print("  2. Click 'Fetch Today's Races'")
        print("  3. Come back and run this test again")
        return False

    # Get most recent race
    most_recent_date = df_races['jour'].max()
    recent_races = df_races[df_races['jour'] == most_recent_date]
    race_to_test = recent_races.iloc[0]
    race_comp = race_to_test['comp']

    print(f"✓ Loaded {len(df_races)} quinté races")
    print(f"  Testing with: {race_comp} on {most_recent_date}\n")

    # Run prediction
    print(f"[3/5] Running prediction (this will show two-stage logs)...")
    print("-" * 80)

    result = predictor.run_prediction(
        race_date=most_recent_date,
        output_dir='predictions/test',
        store_to_db=False
    )

    print("-" * 80)

    if result['status'] != 'success':
        print(f"✗ Prediction failed: {result.get('message')}")
        return False

    print(f"✓ Prediction completed in {result['prediction_time']:.2f}s\n")

    # Analyze results
    print("[4/5] Analyzing results...")
    df_predictions = result['predictions']

    for race_comp in df_predictions['comp'].unique():
        race_df = df_predictions[df_predictions['comp'] == race_comp].copy()
        race_df = race_df.sort_values('predicted_position')

        top5 = race_df.head(5)

        print(f"\n  Race: {race_comp}")
        print(f"  Total horses: {len(race_df)}")

        # Count longshots in top 5
        top5_odds = top5['cotedirect'].values
        longshots = sum((top5_odds >= 15) & (top5_odds <= 35))
        mid_odds = sum((top5_odds >= 10) & (top5_odds <= 18))
        favorites = sum(top5_odds < 10)

        print(f"\n  Top 5 odds distribution:")
        print(f"    Favorites (< 10):     {favorites} horses")
        print(f"    Mid-odds (10-18):     {mid_odds} horses")
        print(f"    Longshots (15-35):    {longshots} horses")

        print(f"\n  Top 5 predictions:")
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            horse_name = row.get('nom', 'Unknown')[:25]
            odds = row.get('cotedirect', 0.0)
            pos = row['predicted_position']

            # Highlight longshots
            if 15 <= odds <= 35:
                marker = "⭐" if idx >= 4 else "  "  # Star for longshot at pos 4-5
            else:
                marker = "  "

            print(f"    {idx}. #{row['numero']:2d} {horse_name:25s} "
                  f"odds {odds:5.1f}  pos {pos:.2f} {marker}")

    # Validation checks
    print("\n[5/5] Validation checks...")
    checks_passed = 0
    total_checks = 4

    # Check 1: No duplicates
    for race_comp in df_predictions['comp'].unique():
        race_df = df_predictions[df_predictions['comp'] == race_comp]
        top5 = race_df.nsmallest(5, 'predicted_position')
        if top5['numero'].duplicated().sum() == 0:
            print("  ✓ No duplicate horses in top 5")
            checks_passed += 1
        else:
            print("  ✗ FAIL: Duplicate horses found!")
        break  # Only check first race

    # Check 2: Exactly 5 horses
    if len(top5) == 5:
        print("  ✓ Exactly 5 horses in top 5")
        checks_passed += 1
    else:
        print(f"  ✗ FAIL: {len(top5)} horses (expected 5)")

    # Check 3: Longshots present
    longshot_count = sum((top5['cotedirect'] >= 15) & (top5['cotedirect'] <= 35))
    if longshot_count >= 1:
        print(f"  ✓ Longshots in top 5: {longshot_count}")
        checks_passed += 1
    else:
        print(f"  ⚠ No longshots in top 5 (this may happen if none in race)")
        checks_passed += 0.5

    # Check 4: position_adjustment column exists
    if 'position_adjustment' in df_predictions.columns:
        print("  ✓ Two-stage adjustments were applied")
        checks_passed += 1
    else:
        print("  ⚠ position_adjustment column not found")

    print(f"\n{'='*80}")
    print(f"RESULT: {checks_passed}/{total_checks} validation checks passed")
    print(f"{'='*80}\n")

    if checks_passed >= 3:
        print("✓ TWO-STAGE PREDICTION IS WORKING CORRECTLY!")
        return True
    else:
        print("⚠ Some validation checks failed - review logs above")
        return False


if __name__ == "__main__":
    success = test_two_stage_with_real_data()
    exit(0 if success else 1)
