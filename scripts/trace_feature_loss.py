#!/usr/bin/env python3
"""
Feature Loss Diagnostic Script

Traces where 45 features are being lost between FeatureCalculator and model prediction.

Expected: 90 features (che_bytype_dnf_rate most important)
Actual: 45 features (all musique features missing)
Result: 0% winner accuracy

Usage:
    python scripts/trace_feature_loss.py --race-id <comp> --date <YYYY-MM-DD>
"""

import sys
import argparse
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.env_setup import AppConfig,get_sqlite_dbpath

from core.calculators.static_feature_calculator import FeatureCalculator
from core.calculators.quinte_feature_calculator import QuinteFeatureCalculator
from utils.model_manager import ModelManager


class FeatureLossTracer:
    """Traces feature loss through the quinté prediction pipeline."""

    def __init__(self):
        """Initialize tracer."""
        self.checkpoints = []
        self.config = AppConfig()
        self.db_path = get_sqlite_dbpath('2years')

    def checkpoint(self, stage: str, df: pd.DataFrame, description: str = ""):
        """Record a checkpoint in the pipeline."""
        feature_cols = [col for col in df.columns if not col.startswith('_')]

        # Categorize features
        musique_features = [col for col in feature_cols if col.startswith(('che_', 'joc_'))]
        equipment_features = [col for col in feature_cols if col.startswith(('blinkers_', 'shoeing_'))]
        basic_features = [col for col in feature_cols if col in ['age', 'dist', 'cotedirect', 'coteprob', 'gainsCarriere']]

        checkpoint_data = {
            'stage': stage,
            'description': description,
            'total_features': len(feature_cols),
            'total_columns': len(df.columns),
            'rows': len(df),
            'musique_features': len(musique_features),
            'equipment_features': len(equipment_features),
            'basic_features': len(basic_features),
            'feature_sample': feature_cols[:10],
            'musique_sample': musique_features[:5],
            'all_features': feature_cols
        }

        self.checkpoints.append(checkpoint_data)

        print(f"\n{'='*80}")
        print(f"CHECKPOINT: {stage}")
        if description:
            print(f"Description: {description}")
        print(f"{'='*80}")
        print(f"DataFrame Shape: ({checkpoint_data['rows']} rows, {checkpoint_data['total_columns']} columns)")
        print(f"Feature Columns: {checkpoint_data['total_features']}")
        print(f"  - Musique features (che_*, joc_*): {checkpoint_data['musique_features']}")
        print(f"  - Equipment features: {checkpoint_data['equipment_features']}")
        print(f"  - Basic features: {checkpoint_data['basic_features']}")
        print(f"Sample features: {checkpoint_data['feature_sample']}")
        if musique_features:
            print(f"Musique sample: {checkpoint_data['musique_sample']}")
        else:
            print(f"⚠️  NO MUSIQUE FEATURES FOUND!")

        return checkpoint_data

    def load_race_from_db(self, race_id: str = None, race_date: str = None):
        """Load a quinté race from database."""
        print(f"\n{'='*80}")
        print(f"LOADING RACE FROM DATABASE")
        print(f"{'='*80}")

        conn = sqlite3.connect(self.db_path)

        if race_id:
            query = "SELECT * FROM daily_race WHERE comp = ? AND quinte = 1"
            df_race = pd.read_sql_query(query, conn, params=(race_id,))
        elif race_date:
            query = "SELECT * FROM daily_race WHERE jour = ? AND quinte = 1 LIMIT 1"
            df_race = pd.read_sql_query(query, conn, params=(race_date,))
        else:
            query = "SELECT * FROM daily_race WHERE quinte = 1 ORDER BY jour DESC LIMIT 1"
            df_race = pd.read_sql_query(query, conn)

        conn.close()

        if len(df_race) == 0:
            print("❌ No quinté race found!")
            return None

        print(f"✓ Loaded race: {df_race['comp'].iloc[0]}")
        print(f"  Date: {df_race['jour'].iloc[0]}")
        print(f"  Hippo: {df_race['hippo'].iloc[0]}")
        print(f"  Prize: {df_race['prixnom'].iloc[0]}")

        return df_race

    def expand_participants(self, df_race: pd.DataFrame) -> pd.DataFrame:
        """Expand partants JSON into individual horse rows."""
        print(f"\n{'='*80}")
        print(f"EXPANDING PARTICIPANTS")
        print(f"{'='*80}")

        partants_json = df_race['participants'].iloc[0]
        partants = json.loads(partants_json) if isinstance(partants_json, str) else partants_json

        print(f"Partants: {len(partants)} horses")

        # Create row for each horse
        horse_rows = []
        for partant in partants:
            horse_row = df_race.iloc[0].to_dict()
            horse_row.update(partant)
            horse_rows.append(horse_row)

        df_participants = pd.DataFrame(horse_rows)

        print(f"✓ Expanded to {len(df_participants)} rows")
        print(f"  Columns: {len(df_participants.columns)}")
        print(f"  Sample horse: numero={df_participants['numero'].iloc[0]}, cheval={df_participants.get('cheval', ['?'])[0] if 'cheval' in df_participants.columns else '?'}")

        # Check for musique column
        if 'musique' in df_participants.columns:
            musique_sample = df_participants['musique'].iloc[0] if pd.notna(df_participants['musique'].iloc[0]) else 'MISSING'
            print(f"  ✓ Musique column present: '{musique_sample}'")
        else:
            print(f"  ⚠️  NO MUSIQUE COLUMN!")

        self.checkpoint("1_EXPAND_PARTICIPANTS", df_participants, "After expanding partants JSON")

        return df_participants

    def calculate_static_features(self, df_participants: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard racing features."""
        print(f"\n{'='*80}")
        print(f"CALCULATING STATIC FEATURES (FeatureCalculator.calculate_all_features)")
        print(f"{'='*80}")

        print(f"Input shape: {df_participants.shape}")
        print(f"Input columns: {len(df_participants.columns)}")

        # Check musique before calculation
        if 'musique' in df_participants.columns:
            musique_count = df_participants['musique'].notna().sum()
            print(f"  Musique present: {musique_count}/{len(df_participants)} horses")
            print(f"  Sample musique: '{df_participants['musique'].iloc[0]}'")
        else:
            print(f"  ⚠️  NO MUSIQUE COLUMN in input!")

        # Call FeatureCalculator
        df_with_features = FeatureCalculator.calculate_all_features(
            df_participants,
            use_temporal=True,
            db_path=self.db_path
        )

        print(f"\nOutput shape: {df_with_features.shape}")
        print(f"Output columns: {len(df_with_features.columns)}")

        # Check for musique features
        musique_features = [col for col in df_with_features.columns if col.startswith(('che_', 'joc_'))]
        print(f"Musique features created: {len(musique_features)}")
        if musique_features:
            print(f"  Sample: {musique_features[:10]}")
        else:
            print(f"  ⚠️  NO MUSIQUE FEATURES CREATED!")

        self.checkpoint("2_STATIC_FEATURES", df_with_features, "After FeatureCalculator.calculate_all_features")

        return df_with_features

    def calculate_quinte_features(self, df_with_static: pd.DataFrame) -> pd.DataFrame:
        """Calculate quinté-specific features."""
        print(f"\n{'='*80}")
        print(f"CALCULATING QUINTÉ FEATURES")
        print(f"{'='*80}")

        quinte_calculator = QuinteFeatureCalculator(self.db_path)

        # Batch-load quinté historical data
        all_quinte_data = quinte_calculator.batch_load_all_quinte_data()
        print(f"Loaded {len(all_quinte_data['races'])} historical quinté races")

        # Process each race
        unique_races = df_with_static['comp'].unique()
        race_dfs = []

        for race_comp in unique_races:
            race_df = df_with_static[df_with_static['comp'] == race_comp].copy()

            # Get race info and date
            race_date = race_df['jour'].iloc[0] if 'jour' in race_df.columns else None
            race_info = {
                'comp': race_df['comp'].iloc[0],
                'hippo': race_df['hippo'].iloc[0] if 'hippo' in race_df.columns else '',
                'dist': race_df['dist'].iloc[0] if 'dist' in race_df.columns else 1600,
                'typec': race_df['typec'].iloc[0] if 'typec' in race_df.columns else 'P'
            }

            # Add quinté features using correct method signature
            # Signature: add_batch_quinte_features(df, race_info, before_date, all_data)
            race_df = quinte_calculator.add_batch_quinte_features(
                race_df,
                race_info,
                race_date,  # before_date parameter
                all_quinte_data
            )

            race_dfs.append(race_df)

        df_with_quinte = pd.concat(race_dfs, ignore_index=True)

        print(f"✓ Quinté features added")

        self.checkpoint("3_QUINTE_FEATURES", df_with_quinte, "After adding quinté features")

        return df_with_quinte

    def load_expected_features(self) -> list:
        """Load expected feature list from model."""
        print(f"\n{'='*80}")
        print(f"LOADING EXPECTED FEATURES FROM MODEL")
        print(f"{'='*80}")

        model_manager = ModelManager()
        tabnet_info = model_manager.load_quinte_model('tabnet')

        expected_features = tabnet_info.get('feature_columns', [])

        print(f"Expected features: {len(expected_features)}")
        print(f"Model path: {tabnet_info['path']}")

        # Categorize expected features
        musique_expected = [f for f in expected_features if f.startswith(('che_', 'joc_'))]
        print(f"  Musique features expected: {len(musique_expected)}")
        if musique_expected:
            print(f"    Sample: {musique_expected[:5]}")

        return expected_features

    def compare_features(self, created_features: list, expected_features: list):
        """Compare created vs expected features."""
        print(f"\n{'='*80}")
        print(f"FEATURE COMPARISON: CREATED VS EXPECTED")
        print(f"{'='*80}")

        created_set = set(created_features)
        expected_set = set(expected_features)

        missing = expected_set - created_set
        extra = created_set - expected_set

        print(f"Created: {len(created_features)} features")
        print(f"Expected: {len(expected_features)} features")
        print(f"Match: {len(created_set & expected_set)} features")
        print(f"Missing: {len(missing)} features")
        print(f"Extra: {len(extra)} features")

        if missing:
            print(f"\n⚠️  MISSING FEATURES ({len(missing)}):")
            musique_missing = [f for f in missing if f.startswith(('che_', 'joc_'))]
            if musique_missing:
                print(f"  Musique features missing: {len(musique_missing)}")
                print(f"    {musique_missing[:10]}")

            equipment_missing = [f for f in missing if f.startswith(('blinkers_', 'shoeing_'))]
            if equipment_missing:
                print(f"  Equipment features missing: {len(equipment_missing)}")
                print(f"    {equipment_missing[:10]}")

            other_missing = [f for f in missing if not f.startswith(('che_', 'joc_', 'blinkers_', 'shoeing_'))]
            if other_missing:
                print(f"  Other features missing: {len(other_missing)}")
                print(f"    {other_missing[:10]}")

        if extra:
            print(f"\n Extra features not in model: {len(extra)}")
            print(f"    {list(extra)[:10]}")

        return {
            'missing': list(missing),
            'extra': list(extra),
            'match': len(created_set & expected_set)
        }

    def analyze_pipeline(self):
        """Analyze where features are lost in the pipeline."""
        print(f"\n{'='*80}")
        print(f"PIPELINE ANALYSIS: FEATURE LOSS DETECTION")
        print(f"{'='*80}")

        if len(self.checkpoints) < 2:
            print("Not enough checkpoints to analyze")
            return

        for i in range(1, len(self.checkpoints)):
            prev = self.checkpoints[i-1]
            curr = self.checkpoints[i]

            feature_delta = curr['total_features'] - prev['total_features']
            musique_delta = curr['musique_features'] - prev['musique_features']

            print(f"\n{prev['stage']} → {curr['stage']}")
            print(f"  Total features: {prev['total_features']} → {curr['total_features']} ({feature_delta:+d})")
            print(f"  Musique features: {prev['musique_features']} → {curr['musique_features']} ({musique_delta:+d})")

            if feature_delta < 0:
                print(f"  ⚠️  FEATURES LOST: {abs(feature_delta)}")

                # Find which features were lost
                prev_features = set(prev['all_features'])
                curr_features = set(curr['all_features'])
                lost = prev_features - curr_features

                if lost:
                    print(f"  Lost features: {list(lost)[:10]}")
            elif feature_delta > 0:
                print(f"  ✓ Features added: {feature_delta}")

    def save_report(self, output_file: str = 'feature_loss_report.json'):
        """Save diagnostic report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'checkpoints': self.checkpoints,
            'summary': {
                'total_checkpoints': len(self.checkpoints),
                'initial_features': self.checkpoints[0]['total_features'] if self.checkpoints else 0,
                'final_features': self.checkpoints[-1]['total_features'] if self.checkpoints else 0,
                'features_lost': self.checkpoints[0]['total_features'] - self.checkpoints[-1]['total_features'] if len(self.checkpoints) > 1 else 0
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved to {output_file}")

    def run_full_trace(self, race_id: str = None, race_date: str = None):
        """Run complete feature loss trace."""
        print(f"\n{'#'*80}")
        print(f"FEATURE LOSS DIAGNOSTIC - FULL TRACE")
        print(f"{'#'*80}")

        # Step 1: Load race
        df_race = self.load_race_from_db(race_id, race_date)
        if df_race is None:
            return

        # Step 2: Expand participants
        df_participants = self.expand_participants(df_race)

        # Step 3: Calculate static features
        df_with_static = self.calculate_static_features(df_participants)

        # Step 4: Calculate quinté features
        df_with_quinte = self.calculate_quinte_features(df_with_static)

        # Step 5: Load expected features
        expected_features = self.load_expected_features()

        # Step 6: Compare
        created_features = [col for col in df_with_quinte.columns if not col.startswith('_')]
        comparison = self.compare_features(created_features, expected_features)

        # Step 7: Analyze pipeline
        self.analyze_pipeline()

        # Step 8: Save report
        self.save_report()

        print(f"\n{'#'*80}")
        print(f"DIAGNOSTIC COMPLETE")
        print(f"{'#'*80}")
        print(f"\nSummary:")
        print(f"  Expected features: {len(expected_features)}")
        print(f"  Created features: {len(created_features)}")
        print(f"  Missing features: {len(comparison['missing'])}")
        print(f"  Match: {comparison['match']} features")

        if comparison['missing']:
            print(f"\n⚠️  CRITICAL: {len(comparison['missing'])} features missing from predictions!")
            print(f"This explains the 0% winner accuracy.")


def main():
    parser = argparse.ArgumentParser(
        description='Trace feature loss in quinté prediction pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--race-id', type=str, help='Specific race comp ID to trace')
    parser.add_argument('--date', type=str, help='Race date (YYYY-MM-DD) to trace')

    args = parser.parse_args()

    tracer = FeatureLossTracer()
    tracer.run_full_trace(race_id=args.race_id, race_date=args.date)


if __name__ == '__main__':
    main()
