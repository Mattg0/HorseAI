#!/usr/bin/env python3
"""
Extract Race Features to JSON

Extracts prepared features for specified races and outputs to JSON format.

Usage:
    # Extract features for 2 specific races
    python race_prediction/extract_race_features.py --races 1613016,1611878

    # Extract features for races on a specific date
    python race_prediction/extract_race_features.py --date 2025-10-08

    # Extract features for latest 2 quinté races
    python race_prediction/extract_race_features.py --latest 2
"""

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import pandas as pd
import numpy as np

from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.calculators.static_feature_calculator import FeatureCalculator
from core.calculators.quinte_feature_calculator import QuinteFeatureCalculator


class RaceFeatureExtractor:
    """
    Extracts prepared features for races.
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = True):
        """Initialize the feature extractor."""
        self.config = AppConfig(config_path)
        self.verbose = verbose

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        self.db_path = get_sqlite_dbpath(self.db_type)

        # Initialize quinté feature calculator
        self.quinte_calculator = QuinteFeatureCalculator(self.db_path)

        self.log_info(f"Initialized RaceFeatureExtractor with database: {self.db_type}")

    def log_info(self, message):
        """Log informational message."""
        if self.verbose:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] {message}")

    def load_races(self, race_comps: Optional[List[str]] = None,
                   race_date: Optional[str] = None,
                   latest_n: Optional[int] = None,
                   quinte_only: bool = False) -> pd.DataFrame:
        """
        Load races from database.

        Args:
            race_comps: List of specific race comp IDs
            race_date: Date to load races from (YYYY-MM-DD)
            latest_n: Number of latest races to load
            quinte_only: Only load quinté races

        Returns:
            DataFrame with race information
        """
        conn = sqlite3.connect(self.db_path)

        # Load from historical_quinte table (matches training pipeline)
        if race_comps:
            self.log_info(f"Loading specific races: {race_comps}")
            placeholders = ','.join(['?' for _ in race_comps])
            query = f"SELECT * FROM historical_quinte WHERE comp IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=race_comps)
        elif race_date:
            self.log_info(f"Loading races for date: {race_date}")
            query = "SELECT * FROM historical_quinte WHERE jour = ?"
            query += " ORDER BY reun, prix"
            df = pd.read_sql_query(query, conn, params=(race_date,))
        elif latest_n:
            self.log_info(f"Loading latest {latest_n} races from historical_quinte")
            query = "SELECT * FROM historical_quinte"
            query += " ORDER BY jour DESC"
            query += f" LIMIT {latest_n}"
            df = pd.read_sql_query(query, conn)
        else:
            raise ValueError("Must specify race_comps, race_date, or latest_n")

        conn.close()

        self.log_info(f"Loaded {len(df)} races from historical_quinte table")
        return df

    def expand_participants(self, df_races: pd.DataFrame) -> pd.DataFrame:
        """
        Expand participants from races DataFrame (matches training pipeline).

        Args:
            df_races: DataFrame with races containing participants JSON

        Returns:
            DataFrame with one row per participant
        """
        self.log_info("Expanding participant data...")

        all_participants = []

        for _, race_row in df_races.iterrows():
            race_data = race_row.to_dict()

            # Try to get participants from JSON column
            participants_json = race_data.get('participants', None)

            if participants_json:
                try:
                    if isinstance(participants_json, str):
                        participants = json.loads(participants_json)
                    else:
                        participants = participants_json

                    for participant in participants:
                        # Combine race-level and participant-level data
                        row = {**race_data, **participant}
                        # Remove the original participants JSON to avoid confusion
                        row.pop('participants', None)
                        all_participants.append(row)

                except (json.JSONDecodeError, TypeError) as e:
                    self.log_info(f"Warning: Could not parse participants for race {race_data.get('comp')}: {e}")
                    continue
            else:
                # No participants JSON - skip this race
                self.log_info(f"Warning: No participants data for race {race_data.get('comp')}")
                continue

        df_participants = pd.DataFrame(all_participants)
        self.log_info(f"Expanded to {len(df_participants)} participant records")

        return df_participants

    def extract_features(self, race_comps: Optional[List[str]] = None,
                        race_date: Optional[str] = None,
                        latest_n: Optional[int] = None,
                        quinte_only: bool = False) -> dict:
        """
        Extract features for specified races.

        Args:
            race_comps: List of specific race comp IDs
            race_date: Date to load races from (YYYY-MM-DD)
            latest_n: Number of latest races to load
            quinte_only: Only load quinté races

        Returns:
            Dict with race data and features
        """
        # Load races
        df_races = self.load_races(race_comps, race_date, latest_n, quinte_only)

        if len(df_races) == 0:
            raise ValueError("No races found")

        # Expand participants from races (matches training pipeline)
        df_participants = self.expand_participants(df_races)

        if len(df_participants) == 0:
            raise ValueError("No participants found")

        # Step 1: Calculate standard racing features
        self.log_info("Calculating standard racing features...")
        df_with_features = FeatureCalculator.calculate_all_features(df_participants)
        self.log_info(f"Standard features calculated: {len(df_with_features.columns)} total columns")

        # Step 2: Batch-load all quinté historical data
        self.log_info("Batch-loading quinté historical data...")
        all_quinte_data = self.quinte_calculator.batch_load_all_quinte_data()
        self.log_info(f"Loaded {len(all_quinte_data['races'])} historical races")

        # Step 3: Add quinté features using batch-loaded data
        self.log_info("Calculating quinté features...")
        unique_races = df_with_features['comp'].unique()

        # Process each race separately to add quinté features
        race_dfs = []
        for race_comp in unique_races:
            race_df = df_with_features[df_with_features['comp'] == race_comp].copy()

            if len(race_df) == 0:
                continue

            # Get race info from first row
            race_info = race_df.iloc[0].to_dict()
            race_date = race_info.get('jour')

            # Add quinté features using batch-loaded data
            try:
                race_df = self.quinte_calculator.add_batch_quinte_features(
                    df=race_df,
                    race_info=race_info,
                    before_date=race_date,
                    all_data=all_quinte_data
                )
                race_dfs.append(race_df)
            except Exception as e:
                self.log_info(f"Warning: Could not add quinté features for race {race_comp}: {e}")
                race_dfs.append(race_df)

        # Combine all races
        df_features = pd.concat(race_dfs, ignore_index=True)
        self.log_info(f"Complete feature set: {len(df_features)} participants, {len(df_features.columns)} features")

        # Get feature columns (exclude metadata)
        metadata_cols = ['comp', 'jour', 'hippo', 'reun', 'prix', 'prixnom',
                        'numero', 'idche', 'age', 'entraineur', 'cotedirect']
        feature_cols = [col for col in df_features.columns if col not in metadata_cols]

        self.log_info(f"Total features extracted: {len(feature_cols)}")

        # Build result structure
        result = {
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'database': self.db_type,
                'total_races': len(df_races),
                'total_participants': len(df_participants),
                'total_features': len(feature_cols)
            },
            'feature_names': feature_cols,
            'races': []
        }

        # Add race data
        for _, race in df_races.iterrows():
            comp = str(race['comp'])
            race_participants = df_features[df_features['comp'] == comp]

            race_data = {
                'comp': comp,
                'date': race['jour'],
                'hippo': race['hippo'],
                'race_id': int(race['prix']),
                'race_name': race['prixnom'],
                'quinte': bool(race.get('quinte', False)),
                'actual_results': race.get('actual_results', None),
                'num_horses': len(race_participants),
                'horses': []
            }

            # Add horse data
            for _, horse in race_participants.iterrows():
                horse_data = {
                    'numero': int(horse['numero']),
                    'idche': int(horse['idche']),
                    'age': int(horse['age']) if pd.notna(horse['age']) else None,
                    'entraineur': horse.get('entraineur', None),
                    'cotedirect': float(horse['cotedirect']) if pd.notna(horse['cotedirect']) else None,
                    'features': {}
                }

                # Add feature values
                for feat in feature_cols:
                    value = horse[feat]
                    # Convert numpy types to Python types for JSON serialization
                    if pd.isna(value):
                        horse_data['features'][feat] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        horse_data['features'][feat] = float(value)
                    elif isinstance(value, np.bool_):
                        horse_data['features'][feat] = bool(value)
                    else:
                        horse_data['features'][feat] = value

                race_data['horses'].append(horse_data)

            result['races'].append(race_data)

        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract race features to JSON format'
    )
    parser.add_argument(
        '--races',
        type=str,
        help='Comma-separated list of race comp IDs (e.g., "1613016,1611878")'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Date to extract races from (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--latest',
        type=int,
        help='Number of latest races to extract'
    )
    parser.add_argument(
        '--quinte-only',
        action='store_true',
        help='Only extract quinté races'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (default: race_features_TIMESTAMP.json)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    args = parser.parse_args()

    # Default to latest 2 races if no arguments provided
    if not any([args.races, args.date, args.latest]):
        args.latest = 2

    # Parse race comps if provided
    race_comps = None
    if args.races:
        race_comps = [comp.strip() for comp in args.races.split(',')]

    # Initialize extractor
    extractor = RaceFeatureExtractor(config_path=args.config)

    # Extract features
    result = extractor.extract_features(
        race_comps=race_comps,
        race_date=args.date,
        latest_n=args.latest,
        quinte_only=args.quinte_only
    )

    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if race_comps:
            race_ids = '_'.join(race_comps[:2])  # Use first 2 race IDs
            output_file = f"race_features_{race_ids}_{timestamp}.json"
        elif args.date:
            output_file = f"race_features_{args.date}_{timestamp}.json"
        else:
            output_file = f"race_features_latest_{timestamp}.json"

    # Save to JSON
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("RACE FEATURES EXTRACTED")
    print("="*80)
    print(f"\nTotal races: {result['metadata']['total_races']}")
    print(f"Total participants: {result['metadata']['total_participants']}")
    print(f"Total features: {result['metadata']['total_features']}")
    print(f"\nRaces extracted:")
    for race in result['races']:
        print(f"  - {race['comp']}: {race['race_name']} ({race['num_horses']} horses)")
    print(f"\nFeature names (first 20):")
    for feat in result['feature_names'][:20]:
        print(f"  - {feat}")
    if len(result['feature_names']) > 20:
        print(f"  ... and {len(result['feature_names']) - 20} more")
    print(f"\nOutput saved to: {output_path.absolute()}")
    print("="*80)


if __name__ == '__main__':
    main()
