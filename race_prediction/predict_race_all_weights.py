#!/usr/bin/env python3
"""
Single Race Multi-Weight Prediction Script

Generates predictions for a single race using configured weight strategies from config.yaml.
Applies all eligible weight configurations based on race metadata (distance, type, field size).
Useful for high-stakes races where you want to see multiple prediction scenarios.

Usage:
    # Generate all eligible predictions for a race
    python race_prediction/predict_race_all_weights.py --race-id 1621325

    # Include all possible configurations (not just matching ones)
    python race_prediction/predict_race_all_weights.py --race-id 1621325 --include-all

    # Save to specific output file
    python race_prediction/predict_race_all_weights.py --race-id 1621325 --output race_1621325_scenarios.csv
"""

import argparse
import sqlite3
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from utils.env_setup import AppConfig, get_sqlite_dbpath


class RaceMultiWeightPredictor:
    """
    Generate all prediction scenarios for a single race using configured weight strategies.

    Loads weight configurations from config.yaml and applies eligible configurations
    based on race metadata (distance, type, field size, etc.).
    """

    def __init__(self, config_path: str = 'config.yaml', verbose: bool = True):
        """Initialize the predictor."""
        self.config = AppConfig(config_path)
        self.verbose = verbose
        self.config_path = config_path

        # Get database configuration
        self.db_type = self.config._config.base.active_db
        self.db_path = get_sqlite_dbpath(self.db_type)

        # Load weight configurations
        self.weight_configs = self._load_weight_configs()

        self.log_info(f"Initialized RaceMultiWeightPredictor with database: {self.db_type}")
        self.log_info(f"Loaded {len(self.weight_configs)} weight configurations from config")

    def log_info(self, message):
        """Log informational message."""
        if self.verbose:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] {message}")

    def _load_weight_configs(self) -> List[Dict]:
        """
        Load weight configurations from config.yaml.

        Returns:
            List of weight configuration dicts
        """
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        configs = []

        # Add default weights
        if 'blend' in config_data and 'default_weights' in config_data['blend']:
            default = config_data['blend']['default_weights']
            configs.append({
                'name': 'default',
                'description': default.get('description', 'Default weights'),
                'condition': {},
                'weights': {
                    'rf_weight': default['rf_weight'],
                    'tabnet_weight': default['tabnet_weight']
                },
                'accuracy': default.get('accuracy')
            })

        # Add dynamic weights
        if 'blend' in config_data and 'dynamic_weights' in config_data['blend']:
            for idx, dynamic in enumerate(config_data['blend']['dynamic_weights']):
                configs.append({
                    'name': f'dynamic_{idx + 1}',
                    'description': dynamic.get('description', f'Dynamic config {idx + 1}'),
                    'condition': dynamic.get('condition', {}),
                    'weights': dynamic['weights'],
                    'accuracy': dynamic.get('accuracy')
                })

        return configs

    def _check_condition_match(self, race_metadata: Dict, condition: Dict) -> bool:
        """
        Check if race metadata matches a condition.

        Args:
            race_metadata: Dict with race metadata (typec, dist, partant, etc.)
            condition: Dict with condition criteria

        Returns:
            True if race matches condition
        """
        if not condition:  # Empty condition matches everything (default)
            return True

        for key, value in condition.items():
            if key.endswith('_min'):
                # Minimum value check
                field = key[:-4]  # Remove '_min' suffix
                if field not in race_metadata:
                    return False
                if race_metadata[field] < value:
                    return False

            elif key.endswith('_max'):
                # Maximum value check
                field = key[:-4]  # Remove '_max' suffix
                if field not in race_metadata:
                    return False
                if race_metadata[field] > value:
                    return False

            else:
                # Exact match check
                if key not in race_metadata:
                    return False
                if race_metadata[key] != value:
                    return False

        return True

    def _get_eligible_configs(self, race_metadata: Dict, include_all: bool = False) -> List[Dict]:
        """
        Get weight configurations eligible for this race.

        Args:
            race_metadata: Dict with race metadata
            include_all: If True, include all configs regardless of match

        Returns:
            List of eligible weight configurations
        """
        if include_all:
            return self.weight_configs

        eligible = []
        for config in self.weight_configs:
            if self._check_condition_match(race_metadata, config['condition']):
                eligible.append(config)

        return eligible

    def load_race_info(self, race_id: str) -> Tuple[Dict, Dict]:
        """
        Load race information and metadata from database.

        Args:
            race_id: Race comp ID

        Returns:
            Tuple of (race_info dict, race_metadata dict)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Try daily_race first for complete info
        query = """
        SELECT comp, jour, hippo, prixnom, actual_results,
               typec, dist, partant
        FROM daily_race
        WHERE comp = ?
        """
        cursor.execute(query, (race_id,))
        row = cursor.fetchone()

        if row:
            race_info = {
                'race_id': row[0],
                'date': row[1],
                'track': row[2],
                'race_name': row[3],
                'actual_results': row[4]
            }
            race_metadata = {
                'typec': row[5],
                'dist': row[6],
                'partant': row[7]
            }
        else:
            # Try quinte_predictions for race info
            query = """
            SELECT DISTINCT race_id, race_date, track, race_name
            FROM quinte_predictions
            WHERE race_id = ?
            """
            cursor.execute(query, (race_id,))
            row = cursor.fetchone()

            if row:
                race_info = {
                    'race_id': row[0],
                    'date': row[1],
                    'track': row[2],
                    'race_name': row[3],
                    'actual_results': None
                }
                race_metadata = {}
            else:
                race_info = {
                    'race_id': race_id,
                    'date': 'Unknown',
                    'track': 'Unknown',
                    'race_name': 'Unknown',
                    'actual_results': None
                }
                race_metadata = {}

        conn.close()

        return race_info, race_metadata

    def load_quinte_predictions(self, race_id: str) -> pd.DataFrame:
        """
        Load quinté model predictions from database.

        Args:
            race_id: Race comp ID

        Returns:
            DataFrame with quinté predictions
        """
        self.log_info(f"Loading quinté predictions for race {race_id}...")

        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            race_id,
            horse_number as numero,
            horse_id as idche,
            horse_name,
            quinte_rf_prediction,
            quinte_tabnet_prediction,
            competitive_adjustment,
            final_prediction,
            predicted_rank
        FROM quinte_predictions
        WHERE race_id = ?
        ORDER BY predicted_rank
        """

        df = pd.read_sql_query(query, conn, params=(race_id,))
        conn.close()

        if len(df) == 0:
            raise ValueError(f"No quinté predictions found for race {race_id}")

        # Fill missing competitive adjustments with 0.0
        df['competitive_adjustment'] = df['competitive_adjustment'].fillna(0.0)

        self.log_info(f"Loaded quinté predictions for {len(df)} horses")

        return df

    def load_general_predictions(self, race_id: str) -> pd.DataFrame:
        """
        Load general model predictions from database.

        Args:
            race_id: Race comp ID

        Returns:
            DataFrame with general model predictions (or empty if not available)
        """
        self.log_info(f"Loading general model predictions for race {race_id}...")

        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            race_id,
            horse_number as numero,
            horse_id as idche,
            final_prediction as general_prediction
        FROM race_predictions
        WHERE race_id = ?
        """

        df = pd.read_sql_query(query, conn, params=(race_id,))
        conn.close()

        if len(df) == 0:
            self.log_info("No general model predictions found")
        else:
            self.log_info(f"Loaded general predictions for {len(df)} horses")

        return df

    def generate_configured_scenarios(self, df_quinte: pd.DataFrame,
                                     weight_configs: List[Dict]) -> List[Dict]:
        """
        Generate prediction scenarios using configured weights from config.yaml.

        Args:
            df_quinte: DataFrame with quinté predictions
            weight_configs: List of weight configuration dicts

        Returns:
            List of scenario dicts with predictions
        """
        scenarios = []

        for config in weight_configs:
            rf_weight = config['weights']['rf_weight']
            tabnet_weight = config['weights']['tabnet_weight']

            # Calculate blended prediction
            df_scenario = df_quinte.copy()
            df_scenario['predicted_position'] = (
                rf_weight * df_scenario['quinte_rf_prediction'] +
                tabnet_weight * df_scenario['quinte_tabnet_prediction']
            )
            df_scenario['predicted_rank'] = df_scenario['predicted_position'].rank(method='first').astype(int)
            df_scenario = df_scenario.sort_values('predicted_rank')

            # Get top 5
            top5 = df_scenario.head(5)[['numero', 'horse_name', 'predicted_position', 'predicted_rank']].to_dict('records')

            # Build description
            description = config['description']
            if config.get('accuracy'):
                description += f" (Accuracy: {config['accuracy']:.1f}%)"

            scenarios.append({
                'scenario_type': 'Configured_Blend',
                'config_name': config['name'],
                'rf_weight': round(rf_weight, 2),
                'tabnet_weight': round(tabnet_weight, 2),
                'description': description,
                'condition': config['condition'],
                'accuracy': config.get('accuracy'),
                'top5': top5,
                'all_predictions': df_scenario[['numero', 'horse_name', 'predicted_position', 'predicted_rank']].to_dict('records')
            })

        return scenarios

    def predict_race(self, race_id: str, include_all: bool = False) -> Dict:
        """
        Generate all prediction scenarios for a race using configured weights.

        Args:
            race_id: Race comp ID
            include_all: If True, include all weight configs regardless of match

        Returns:
            Dict with all scenarios
        """
        self.log_info("=" * 80)
        self.log_info(f"GENERATING PREDICTION SCENARIOS FOR RACE {race_id}")
        self.log_info("=" * 80)

        # Load race info and metadata
        race_info, race_metadata = self.load_race_info(race_id)
        self.log_info(f"\nRace: {race_info['race_name']}")
        self.log_info(f"Date: {race_info['date']}")
        self.log_info(f"Track: {race_info['track']}")

        if race_metadata:
            self.log_info(f"Type: {race_metadata.get('typec', 'Unknown')}")
            self.log_info(f"Distance: {race_metadata.get('dist', 'Unknown')}m")
            self.log_info(f"Field size: {race_metadata.get('partant', 'Unknown')} horses")

        # Load predictions
        df_quinte = self.load_quinte_predictions(race_id)

        # Get eligible weight configurations
        eligible_configs = self._get_eligible_configs(race_metadata, include_all)
        self.log_info(f"\n{len(eligible_configs)} eligible weight configurations found")

        if not eligible_configs:
            self.log_info("⚠️  No matching weight configurations - using default only")
            eligible_configs = [config for config in self.weight_configs if config['name'] == 'default']

        # Generate scenarios
        self.log_info("\nGenerating prediction scenarios with configured weights...")
        scenarios = self.generate_configured_scenarios(df_quinte, eligible_configs)

        # Sort by accuracy (if available)
        scenarios_sorted = sorted(scenarios, key=lambda x: x.get('accuracy') or 0, reverse=True)

        # Compile results
        results = {
            'race_info': race_info,
            'race_metadata': race_metadata,
            'num_horses': len(df_quinte),
            'eligible_configs': len(eligible_configs),
            'scenarios': scenarios_sorted,
            'total_scenarios': len(scenarios_sorted)
        }

        self.log_info(f"\n✓ Generated {results['total_scenarios']} prediction scenarios")

        return results

    def save_results(self, results: Dict, output_file: str):
        """
        Save results to CSV and JSON files.

        Args:
            results: Results dict from predict_race()
            output_file: Output file path (without extension)
        """
        output_path = Path(output_file)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed JSON
        json_file = output_path.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        self.log_info(f"✓ Saved detailed results to {json_file}")

        # Create CSV with top 5 for each scenario
        csv_data = []
        race_info = results['race_info']
        race_metadata = results['race_metadata']

        for scenario in results['scenarios']:
            for rank, horse in enumerate(scenario['top5'], 1):
                row = {
                    'race_id': race_info['race_id'],
                    'race_name': race_info['race_name'],
                    'date': race_info['date'],
                    'track': race_info['track'],
                    'race_type': race_metadata.get('typec', 'Unknown'),
                    'distance': race_metadata.get('dist', None),
                    'field_size': race_metadata.get('partant', None),
                    'config_name': scenario['config_name'],
                    'scenario_description': scenario['description'],
                    'rf_weight': scenario['rf_weight'],
                    'tabnet_weight': scenario['tabnet_weight'],
                    'config_accuracy': scenario.get('accuracy'),
                    'predicted_rank': rank,
                    'horse_number': horse['numero'],
                    'horse_name': horse['horse_name'],
                    'predicted_position': round(horse['predicted_position'], 3)
                }

                csv_data.append(row)

        df_csv = pd.DataFrame(csv_data)
        csv_file = output_path.with_suffix('.csv')
        df_csv.to_csv(csv_file, index=False)
        self.log_info(f"✓ Saved CSV summary to {csv_file}")

    def print_summary(self, results: Dict):
        """
        Print a summary of all scenarios.

        Args:
            results: Results dict from predict_race()
        """
        race_info = results['race_info']
        race_metadata = results['race_metadata']

        print("\n" + "=" * 80)
        print("PREDICTION SCENARIOS SUMMARY")
        print("=" * 80)
        print(f"\nRace: {race_info['race_name']}")
        print(f"Date: {race_info['date']}")
        print(f"Track: {race_info['track']}")

        if race_metadata:
            print(f"Type: {race_metadata.get('typec', 'Unknown')}")
            print(f"Distance: {race_metadata.get('dist', 'Unknown')}m")
            print(f"Field Size: {race_metadata.get('partant', 'Unknown')}")

        print(f"Horses: {results['num_horses']}")
        print(f"Eligible Weight Configurations: {results['eligible_configs']}")
        print(f"Total Scenarios: {results['total_scenarios']}")

        if race_info.get('actual_results'):
            print(f"Actual Results: {race_info['actual_results']}")

        # Print all scenarios (sorted by accuracy)
        print(f"\n{'-' * 80}")
        print("CONFIGURED WEIGHT SCENARIOS (sorted by historical accuracy)")
        print(f"{'-' * 80}")

        for idx, scenario in enumerate(results['scenarios'], 1):
            is_matching = self._check_condition_match(race_metadata, scenario['condition']) if race_metadata else False
            match_indicator = " ✓ MATCH" if is_matching else ""

            print(f"\n{idx}. {scenario['description']}{match_indicator}")
            print(f"   Weights: RF={scenario['rf_weight']}, TabNet={scenario['tabnet_weight']}")
            if scenario['condition']:
                cond_str = ', '.join([f"{k}={v}" for k, v in scenario['condition'].items()])
                print(f"   Condition: {cond_str}")
            print(f"   Top 5: {', '.join([str(h['numero']) for h in scenario['top5']])}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate prediction scenarios for a single race using configured weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate eligible scenarios for a race (only matching weight configs)
  python race_prediction/predict_race_all_weights.py --race-id 1621325

  # Include all weight configurations (not just matching ones)
  python race_prediction/predict_race_all_weights.py --race-id 1621325 --include-all

  # Save to specific output file
  python race_prediction/predict_race_all_weights.py --race-id 1621325 --output scenarios/race_1621325
        """
    )
    parser.add_argument(
        '--race-id',
        type=str,
        required=True,
        help='Race comp ID to generate predictions for'
    )
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Include all weight configurations (not just matching ones)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (without extension). If not specified, uses race_scenarios_{race_id}_{timestamp}'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = RaceMultiWeightPredictor(config_path=args.config)

    # Generate predictions
    results = predictor.predict_race(
        race_id=args.race_id,
        include_all=args.include_all
    )

    # Print summary
    predictor.print_summary(results)

    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"predictions/race_scenarios_{args.race_id}_{timestamp}"

    predictor.save_results(results, output_file)

    print(f"\n{'=' * 80}")
    print(f"✓ All scenarios saved!")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
