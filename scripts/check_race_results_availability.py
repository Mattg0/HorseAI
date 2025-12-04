#!/usr/bin/env python3
"""
Quick diagnostic to check what race results are available for comparison.

This script doesn't require any heavy dependencies - just sqlite3.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

def check_available_results():
    """Check what race results are available in the database."""

    # Find database
    db_candidates = [
        'data/hippique2.db',
        'data/hippique5.db',
        'data/hippique.db',
        'hippique2.db'
    ]

    db_path = None
    for candidate in db_candidates:
        if Path(candidate).exists():
            db_path = candidate
            break

    if not db_path:
        print("✗ No database found")
        return

    print(f"Using database: {db_path}\n")

    conn = sqlite3.connect(db_path)

    # Check quinté races with actual results
    query = """
    SELECT
        comp,
        jour,
        hippo,
        prixnom,
        actual_results,
        prediction_results IS NOT NULL as has_predictions
    FROM daily_race
    WHERE quinte = 1
        AND actual_results IS NOT NULL
        AND actual_results != ''
    ORDER BY jour DESC
    LIMIT 50
    """

    cursor = conn.execute(query)
    results = cursor.fetchall()

    if not results:
        print("✗ No quinté races with actual results found")
        print("\nTo populate results:")
        print("  1. Run races through UIApp")
        print("  2. Fetch actual results from PMU API")
        conn.close()
        return

    print(f"✓ Found {len(results)} quinté races with actual results\n")
    print("="*100)
    print(f"{'Date':<12} {'Race':<15} {'Track':<20} {'Results':<30} {'Has Pred'}")
    print("="*100)

    races_with_both = 0
    races_needing_pred = 0

    for row in results:
        comp, jour, hippo, prixnom, actual_results, has_predictions = row

        # Format results
        if actual_results and len(str(actual_results)) > 30:
            results_str = str(actual_results)[:27] + "..."
        else:
            results_str = str(actual_results)

        # Format race name
        if prixnom and len(prixnom) > 20:
            prixnom = prixnom[:17] + "..."

        pred_status = "✓ Yes" if has_predictions else "✗ No"

        print(f"{jour:<12} {comp:<15} {hippo[:19]:<20} {results_str:<30} {pred_status}")

        if has_predictions:
            races_with_both += 1
        else:
            races_needing_pred += 1

    conn.close()

    print("="*100)
    print(f"\nSummary:")
    print(f"  Total quinté races with results: {len(results)}")
    print(f"  Races with both predictions & results: {races_with_both}")
    print(f"  Races needing predictions: {races_needing_pred}")

    if races_with_both > 0:
        print(f"\n✓ You can run comparison on {races_with_both} races!")
        print(f"\nTo compare standard vs two-stage predictions:")
        print(f"  python3 scripts/compare_standard_vs_twostage.py")
    else:
        print(f"\n⚠ Need to generate predictions first")
        print(f"  1. Open UIApp: streamlit run UI/UIApp.py")
        print(f"  2. Go to Quinté Predictions tab")
        print(f"  3. Run predictions on these dates")
        print(f"  4. Then run: python3 scripts/compare_standard_vs_twostage.py")

if __name__ == "__main__":
    print("\n" + "="*100)
    print("CHECKING RACE RESULTS AVAILABILITY FOR COMPARISON")
    print("="*100 + "\n")

    try:
        check_available_results()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
