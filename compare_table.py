#!/usr/bin/env python
"""
Simple script to display blending comparison results from results.json in a tabular format.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate


def process_results_json(results_file_path):
    """
    Process the results.json file and display as a formatted table.

    Args:
        results_file_path: Path to the results.json file
    """
    # Load the results file
    try:
        with open(results_file_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error reading results file: {e}")
        return

    # Create a list to store the data for the table
    table_data = []

    # Sort blending values numerically
    blend_values = sorted([float(k) for k in results.keys()])

    # Process each blending value
    for blend_value in blend_values:
        blend_str = str(blend_value)
        if blend_str in results:
            summary = results[blend_str].get('summary', {})

            # Extract key metrics
            row_data = {
                'Blend Value': f"{blend_value:.2f}",
                'Winner Acc': f"{summary.get('winner_accuracy', 0):.4f}",
                'Podium Acc': f"{summary.get('podium_accuracy', 0):.4f}",
                'Win Count': summary.get('winner_count', 0),
                'Top3 Count': summary.get('top3_count', 0),
                'Races': summary.get('total_with_metrics', 0),
                'Tiercé Exact': f"{summary.get('pmu_bets', {}).get('tierce_exact_rate', 0):.4f}",
                'Tiercé Désordre': f"{summary.get('pmu_bets', {}).get('tierce_desordre_rate', 0):.4f}",
                'Quarté Exact': f"{summary.get('pmu_bets', {}).get('quarte_exact_rate', 0):.4f}",
                'Quinte Exact': f"{summary.get('pmu_bets', {}).get('quinte_exact_rate', 0):.4f}",
                'Arriv':
            }

            table_data.append(row_data)

    # Convert to pandas DataFrame for easier display
    if table_data:
        df = pd.DataFrame(table_data)

        # Print the table
        print("\n=== BLENDING COMPARISON RESULTS ===\n")
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

        # Find and print the best values
        print("\n=== BEST VALUES ===\n")
        best_values = {}

        # Numeric columns that we want to find the maximum for
        numeric_cols = [
            'Winner Acc', 'Podium Acc', 'Win Count', 'Top3 Count',
            'Tiercé Exact', 'Tiercé Désordre', 'Quarté Exact', 'Quinte Exact'
        ]

        for col in numeric_cols:
            # Convert to numeric for comparison
            if col in ['Win Count', 'Top3 Count', 'Races']:
                # These are already integers
                values = [int(row[col]) for row in table_data]
            else:
                # Convert from string format to float
                values = [float(row[col]) for row in table_data]

            # Find the max value and corresponding blend value
            max_value = max(values)
            max_index = values.index(max_value)
            max_blend = table_data[max_index]['Blend Value']

            best_values[col] = (max_value, max_blend)

        # Create a dataframe for best values
        best_df = pd.DataFrame({
            'Metric': list(best_values.keys()),
            'Best Value': [f"{v[0]:.4f}" if isinstance(v[0], float) else v[0] for v in best_values.values()],
            'Best Blend': [v[1] for v in best_values.values()]
        })

        print(tabulate(best_df, headers='keys', tablefmt='grid', showindex=False))

    else:
        print("No data found in results file.")


def main():
    parser = argparse.ArgumentParser(description="Display blending comparison results in a table")
    parser.add_argument("results_file", type=str, help="Path to results.json file")

    args = parser.parse_args()

    # Process the results file
    process_results_json(args.results_file)


if __name__ == "__main__":
    main()