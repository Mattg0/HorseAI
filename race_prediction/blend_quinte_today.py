#!/usr/bin/env python3
"""
Manual Quinté Blend Script for Today's Predictions

Blends today's quinté predictions (from JSON) with general model predictions (from DB)
using optimal weights: 0.20 quinté, 0.80 general

Usage:
    python race_prediction/blend_quinte_today.py
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from utils.env_setup import AppConfig, get_sqlite_dbpath


def load_latest_quinte_predictions(predictions_dir='predictions'):
    """Load the most recent quinté predictions JSON file."""
    pred_path = Path(predictions_dir)
    json_files = list(pred_path.glob("quinte_predictions_*.json"))

    if not json_files:
        raise FileNotFoundError(f"No quinté prediction files found in {predictions_dir}")

    # Get most recent file
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)

    print(f"Loading quinté predictions from: {latest_file}")

    with open(latest_file, 'r') as f:
        data = json.load(f)

    # Flatten to DataFrame
    all_predictions = []
    for race in data:
        for pred in race['predictions']:
            all_predictions.append(pred)

    df = pd.DataFrame(all_predictions)

    # Rename prediction column
    if 'predicted_position' in df.columns:
        df['quinte_prediction'] = df['predicted_position']
    elif 'predicted_position_tabnet' in df.columns:
        df['quinte_prediction'] = df['predicted_position_tabnet']

    print(f"Loaded {len(df)} quinté predictions for {df['comp'].nunique()} races")

    return df


def load_general_predictions(race_comps, db_path):
    """Load general model predictions from race_predictions table."""
    print(f"\nLoading general model predictions from database...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    race_comps_str = [str(comp) for comp in race_comps]
    placeholders = ','.join(['?' for _ in race_comps_str])

    query = f"""
    SELECT race_id, horse_id, final_prediction
    FROM race_predictions
    WHERE race_id IN ({placeholders})
    """

    cursor.execute(query, race_comps_str)
    rows = cursor.fetchall()
    conn.close()

    # Create mapping
    predictions = []
    for race_id, horse_id, final_pred in rows:
        predictions.append({
            'comp': str(race_id),
            'idche': int(horse_id),
            'general_prediction': float(final_pred)
        })

    df_general = pd.DataFrame(predictions)

    print(f"Loaded general predictions for {df_general['comp'].nunique()} races")

    return df_general


def blend_predictions(df_quinte, df_general, quinte_weight=0.20, general_weight=0.80):
    """Blend quinté and general predictions."""
    print(f"\nBlending predictions with weights: Quinté={quinte_weight:.2f}, General={general_weight:.2f}")

    # Merge on race comp and horse ID
    df_merged = df_quinte.merge(
        df_general,
        on=['comp', 'idche'],
        how='left'
    )

    # Count matches
    matched = df_merged['general_prediction'].notna().sum()
    total = len(df_merged)

    print(f"Matched {matched}/{total} horses with general predictions")

    # Only keep races where we have both predictions
    df_with_both = df_merged[df_merged['general_prediction'].notna()].copy()

    # Blend predictions
    df_with_both['blended_prediction'] = (
        quinte_weight * df_with_both['quinte_prediction'] +
        general_weight * df_with_both['general_prediction']
    )

    # Calculate rank within each race
    df_with_both['predicted_rank'] = df_with_both.groupby('comp')['blended_prediction'].rank(method='first')

    print(f"Blended predictions for {df_with_both['comp'].nunique()} races")

    return df_with_both


def format_race_predictions(df_blended):
    """Format predictions by race."""
    results = []

    for comp, race_df in df_blended.groupby('comp'):
        race_df_sorted = race_df.sort_values('predicted_rank')

        # Get race info
        jour = race_df_sorted['jour'].iloc[0] if 'jour' in race_df_sorted.columns else 'N/A'
        hippo = race_df_sorted['hippo'].iloc[0] if 'hippo' in race_df_sorted.columns else 'N/A'
        prix = race_df_sorted['prixnom'].iloc[0] if 'prixnom' in race_df_sorted.columns else 'N/A'

        # Get top 5
        top5 = race_df_sorted.head(5)
        top5_numeros = top5['numero'].tolist()
        top5_str = '-'.join([str(n) for n in top5_numeros])

        # Get all predictions
        all_predictions = []
        for _, row in race_df_sorted.iterrows():
            all_predictions.append({
                'rank': int(row['predicted_rank']),
                'numero': int(row['numero']),
                'nom': row.get('nom', ''),
                'quinte_pred': float(row['quinte_prediction']),
                'general_pred': float(row['general_prediction']),
                'blended_pred': float(row['blended_prediction'])
            })

        results.append({
            'comp': comp,
            'jour': jour,
            'hippo': hippo,
            'prix': prix,
            'top5': top5_str,
            'top5_list': top5_numeros,
            'predictions': all_predictions
        })

    return results


def print_results(results):
    """Print formatted results."""
    print("\n" + "="*100)
    print("BLENDED QUINTÉ PREDICTIONS (0.20 Quinté + 0.80 General)")
    print("="*100)

    for race in results:
        print(f"\n{'='*100}")
        print(f"Race: {race['comp']}")
        print(f"Date: {race['jour']}")
        print(f"Venue: {race['hippo']}")
        print(f"Prix: {race['prix']}")
        print(f"{'='*100}")

        print(f"\n🎯 TOP 5 PREDICTION: {race['top5']}")

        print(f"\n{'Rank':<6} {'Num':<6} {'Horse':<25} {'Quinté':<10} {'General':<10} {'Blended':<10}")
        print("-"*100)

        for pred in race['predictions']:
            rank = pred['rank']
            numero = pred['numero']
            nom = pred['nom'][:24]  # Truncate long names
            quinte = pred['quinte_pred']
            general = pred['general_pred']
            blended = pred['blended_pred']

            # Highlight top 5
            marker = "🏆" if rank <= 5 else "  "

            print(f"{marker} {rank:<4} {numero:<6} {nom:<25} {quinte:<10.3f} {general:<10.3f} {blended:<10.3f}")

    print("\n" + "="*100)


def main():
    """Main execution."""
    print("="*100)
    print("QUINTÉ PREDICTION BLENDER")
    print("="*100)

    # Get database path
    config = AppConfig()
    db_type = config._config.base.active_db
    db_path = get_sqlite_dbpath(db_type)

    print(f"Database: {db_type}")

    # Load quinté predictions
    df_quinte = load_latest_quinte_predictions()

    # Get race comps
    race_comps = df_quinte['comp'].unique().tolist()

    # Load general predictions
    df_general = load_general_predictions(race_comps, db_path)

    # Blend predictions
    df_blended = blend_predictions(df_quinte, df_general)

    # Format results
    results = format_race_predictions(df_blended)

    # Print results
    print_results(results)

    # Save to file
    output_file = f"predictions/blended_quinte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {output_file}")
    print("="*100)


if __name__ == '__main__':
    main()
