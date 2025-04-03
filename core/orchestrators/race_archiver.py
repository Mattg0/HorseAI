#!/usr/bin/env python
# data_management/race_archiver.py

import os
import sys
import sqlite3
import json
import pandas as pd
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.env_setup import AppConfig, get_sqlite_dbpath


class RaceArchiver:
    """
    Archives completed races from daily_race table to historical_races table.
    Handles data transformation, validation and cleanup.
    """

    def __init__(self, db_name: str = None, verbose: bool = False):
        """
        Initialize the race archiver.

        Args:
            db_name: Database name from config (defaults to active_db)
            verbose: Whether to enable verbose output
        """
        # Initialize config
        self.config = AppConfig()

        # Set database
        if db_name is None:
            self.db_name = self.config._config.base.active_db
        else:
            self.db_name = db_name

        # Get database path
        self.db_path = self.config.get_sqlite_dbpath(self.db_name)

        # Set verbosity
        self.verbose = verbose

        # Initialize logging
        self._setup_logging()

        # Create necessary database tables
        self._ensure_database_structure()

    def _setup_logging(self):
        """Set up logging."""
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"race_archiver_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )

        self.logger = logging.getLogger("RaceArchiver")
        self.logger.info(f"Logging initialized to {log_file}")

    def _ensure_database_structure(self):
        """Ensure the database has the required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if historical_races table exists, create if not
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_races (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comp VARCHAR(50) NOT NULL,          -- Race identifier
            jour DATE NOT NULL,                 -- Race date
            reunion VARCHAR(20),                -- Meeting number (reun)
            prix VARCHAR(20),                   -- Race number
            quinte BOOLEAN,                     -- Whether it's a Quinte+ race
            hippo VARCHAR(100) NOT NULL,        -- Racecourse
            meteo VARCHAR(100),                 -- Weather
            dist INTEGER,                       -- Distance in meters
            corde VARCHAR(50),                  -- Rail position
            natpis VARCHAR(50),                 -- Track surface
            pistegp VARCHAR(50),                -- Track condition
            typec VARCHAR(20),                  -- Race type
            partant INTEGER,                    -- Number of runners
            temperature REAL,                   -- Temperature
            forceVent REAL,                     -- Wind force
            directionVent VARCHAR(50),          -- Wind direction
            nebulosite VARCHAR(100),            -- Cloud cover
            participants JSON,                  -- Participant data with calculated features
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create race_results table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS race_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comp VARCHAR(50) NOT NULL,          -- Race identifier
            ordre_arrivee JSON NOT NULL,        -- JSON with arrival order
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(comp)
        )
        """)

        # Create indices for faster querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_races_comp ON historical_races(comp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_races_jour ON historical_races(jour)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_race_results_comp ON race_results(comp)")

        # Check for archived_races table (to track what we've archived)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS archived_races (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comp VARCHAR(50) NOT NULL,          -- Race identifier
            archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(comp)
        )
        """)

        conn.commit()
        conn.close()

        self.logger.info("Database structure verified")

    def get_completed_races(self, date_from: str = None, date_to: str = None,
                            limit: int = None, force: bool = False) -> List[Dict]:
        """
        Get completed races that are ready for archiving.

        Args:
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            limit: Maximum number of races to retrieve
            force: Include already archived races

        Returns:
            List of race dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query with optional date filters
        query = """
        SELECT dr.* FROM daily_race dr
        WHERE dr.actual_results IS NOT NULL 
        AND dr.actual_results != 'pending'
        """

        # Skip already archived races unless force is True
        if not force:
            query += """
            AND NOT EXISTS (
                SELECT 1 FROM archived_races ar 
                WHERE ar.comp = dr.comp
            )
            """

        params = []
        if date_from:
            query += " AND dr.jour >= ?"
            params.append(date_from)
        if date_to:
            query += " AND dr.jour <= ?"
            params.append(date_to)

        query += " ORDER BY dr.jour DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to list of dicts and parse JSON fields
        races = []
        for row in rows:
            race_dict = dict(row)

            # Parse JSON fields
            for field in ['participants', 'prediction_results', 'actual_results']:
                if race_dict.get(field):
                    try:
                        if isinstance(race_dict[field], str):
                            race_dict[field] = json.loads(race_dict[field])
                    except json.JSONDecodeError:
                        self.logger.warning(f"Could not parse {field} for race {race_dict['comp']}")
                    except Exception as e:
                        self.logger.warning(f"Error processing {field}: {str(e)}")

            races.append(race_dict)

        conn.close()
        self.logger.info(f"Found {len(races)} completed races for archiving")
        return races

    def transform_race_for_archiving(self, race: Dict) -> Tuple[Dict, Dict]:
        """
        Transform daily race format to historical races format.

        Args:
            race: Dictionary with daily race data

        Returns:
            Tuple of (race_data, results_data)
        """
        # Extract base race information
        historical_race = {
            'comp': race['comp'],
            'jour': race['jour'],
            'reunion': race['reun'],  # daily_race uses 'reun', historical uses 'reunion'
            'prix': race['prix'],
            'quinte': bool(race['quinte']),
            'hippo': race['hippo'],
            'meteo': race['meteo'],
            'dist': race['dist'],
            'corde': race['corde'],
            'natpis': race['natpis'],
            'pistegp': race.get('pistegp'),
            'typec': race['typec'],
            'partant': race['partant'],
            'temperature': race['temperature'],
            'forceVent': race['forceVent'],
            'directionVent': race['directionVent'],
            'nebulosite': race['nebulosite'],
            'created_at': datetime.now().isoformat()
        }

        # Extract participant data
        participants_data = race.get('participants', [])

        # Make sure participants is a list
        if isinstance(participants_data, str):
            try:
                participants_data = json.loads(participants_data)
            except:
                participants_data = []

        # Ensure it's a list
        if not isinstance(participants_data, list):
            participants_data = []

        # Convert to JSON string
        historical_race['participants'] = json.dumps(participants_data)

        # Extract race results
        results_data = None
        actual_results = race.get('actual_results')

        if actual_results:
            # Handle different result formats
            ordre_arrivee = []

            if isinstance(actual_results, str) and '-' in actual_results:
                # Format: "1-4-2-3"
                numeros = actual_results.split('-')
                ordre_arrivee = [
                    {'narrivee': i + 1, 'cheval': int(numeros[i])}
                    for i in range(len(numeros))
                ]
            elif isinstance(actual_results, list):
                # Format: [{"numero": "1", "position": "1"}, ...]
                for result in actual_results:
                    if 'numero' in result and ('position' in result or 'narrivee' in result):
                        position = result.get('position', result.get('narrivee'))

                        try:
                            position = int(position)
                        except (ValueError, TypeError):
                            continue

                        ordre_arrivee.append({
                            'narrivee': position,
                            'cheval': int(result['numero'])
                        })
            elif isinstance(actual_results, dict):
                # Format: {"arrivee": "1-4-2-3"}
                if 'arrivee' in actual_results:
                    numeros = actual_results['arrivee'].split('-')
                    ordre_arrivee = [
                        {'narrivee': i + 1, 'cheval': int(numeros[i])}
                        for i in range(len(numeros))
                    ]

            # Sort by position
            ordre_arrivee.sort(key=lambda x: x['narrivee'])

            # Create results data
            results_data = {
                'comp': race['comp'],
                'ordre_arrivee': json.dumps(ordre_arrivee),
                'created_at': datetime.now().isoformat()
            }

        return historical_race, results_data

    def archive_race(self, race: Dict) -> bool:
        """
        Archive a race to historical tables.

        Args:
            race: Dictionary with race data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Transform race for archiving
            historical_race, results_data = self.transform_race_for_archiving(race)

            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert into historical_races
            historical_cols = ', '.join(historical_race.keys())
            historical_placeholders = ', '.join(['?' for _ in historical_race.keys()])

            cursor.execute(
                f"INSERT OR REPLACE INTO historical_races ({historical_cols}) VALUES ({historical_placeholders})",
                list(historical_race.values())
            )

            # Insert into race_results if we have results
            if results_data:
                results_cols = ', '.join(results_data.keys())
                results_placeholders = ', '.join(['?' for _ in results_data.keys()])

                cursor.execute(
                    f"INSERT OR REPLACE INTO race_results ({results_cols}) VALUES ({results_placeholders})",
                    list(results_data.values())
                )

            # Mark as archived
            cursor.execute(
                "INSERT OR REPLACE INTO archived_races (comp) VALUES (?)",
                (race['comp'],)
            )

            conn.commit()
            conn.close()

            self.logger.info(f"Successfully archived race {race['comp']}")
            return True

        except Exception as e:
            self.logger.error(f"Error archiving race {race.get('comp')}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def archive_races(self, races: List[Dict]) -> Dict[str, Any]:
        """
        Archive multiple races.

        Args:
            races: List of race dictionaries

        Returns:
            Dictionary with results
        """
        start_time = datetime.now()

        successful = 0
        failed = 0
        already_archived = 0

        results = []

        for race in races:
            comp = race.get('comp')

            # Check if already archived
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM archived_races WHERE comp = ?",
                (comp,)
            )
            is_archived = cursor.fetchone() is not None
            conn.close()

            if is_archived:
                self.logger.info(f"Race {comp} already archived, skipping")
                already_archived += 1
                results.append({
                    'comp': comp,
                    'status': 'skipped',
                    'reason': 'already_archived'
                })
                continue

            # Archive the race
            if self.archive_race(race):
                successful += 1
                results.append({
                    'comp': comp,
                    'status': 'success'
                })
            else:
                failed += 1
                results.append({
                    'comp': comp,
                    'status': 'failed'
                })

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        summary = {
            'total': len(races),
            'successful': successful,
            'failed': failed,
            'already_archived': already_archived,
            'execution_time': execution_time,
            'results': results
        }

        self.logger.info(f"Archived {successful} races, {failed} failed, {already_archived} already archived")
        return summary

    def clean_archived_races(self, older_than_days: int = 30, dry_run: bool = True) -> Dict[str, Any]:
        """
        Remove daily races that have been archived and are older than specified days.

        Args:
            older_than_days: Remove races older than this many days
            dry_run: If True, only report what would be removed without actually deleting

        Returns:
            Dictionary with cleanup results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate cutoff date
        cutoff_date = (datetime.now() - timedelta(days=older_than_days)).strftime('%Y-%m-%d')

        # Find races to delete
        query = """
        SELECT dr.comp, dr.jour, dr.hippo, dr.prix FROM daily_race dr
        JOIN archived_races ar ON dr.comp = ar.comp
        WHERE dr.jour < ?
        """

        cursor.execute(query, (cutoff_date,))
        to_delete = cursor.fetchall()

        races_info = []
        for race in to_delete:
            races_info.append({
                'comp': race[0],
                'jour': race[1],
                'hippo': race[2],
                'prix': race[3]
            })

        # Actually delete if not dry run
        if not dry_run and to_delete:
            delete_query = """
            DELETE FROM daily_race
            WHERE comp IN (
                SELECT dr.comp FROM daily_race dr
                JOIN archived_races ar ON dr.comp = ar.comp
                WHERE dr.jour < ?
            )
            """

            cursor.execute(delete_query, (cutoff_date,))
            conn.commit()

            self.logger.info(f"Deleted {len(to_delete)} archived races older than {cutoff_date}")

        conn.close()

        return {
            'mode': 'dry_run' if dry_run else 'delete',
            'cutoff_date': cutoff_date,
            'count': len(to_delete),
            'races': races_info
        }

    def run_archive_pipeline(self, date_from: str = None, date_to: str = None,
                             limit: int = None, clean: bool = False,
                             clean_days: int = 30, dry_run: bool = True) -> Dict[str, Any]:
        """
        Run the full archiving pipeline.

        Args:
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            limit: Maximum number of races to archive
            clean: Whether to clean up old archived races
            clean_days: Remove races older than this many days
            dry_run: If True, only report what would be removed without actually deleting

        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()

        # 1. Get completed races
        races = self.get_completed_races(date_from, date_to, limit)

        if not races:
            return {
                'status': 'no_races',
                'message': 'No races found for archiving'
            }

        # 2. Archive races
        archive_results = self.archive_races(races)

        # 3. Clean up old races if requested
        if clean:
            cleanup_results = self.clean_archived_races(clean_days, dry_run)
        else:
            cleanup_results = {
                'status': 'skipped',
                'message': 'Cleanup was disabled'
            }

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Compile results
        pipeline_results = {
            'status': 'success',
            'execution_time': execution_time,
            'archive_results': archive_results,
            'cleanup_results': cleanup_results
        }

        return pipeline_results


def main():
    """Command-line interface for race archiving."""
    parser = argparse.ArgumentParser(description='Archive daily races to historical database')
    parser.add_argument('--db', help='Database name from config (defaults to active_db)')
    parser.add_argument('--from-date', help='Start date for archiving (YYYY-MM-DD)')
    parser.add_argument('--to-date', help='End date for archiving (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, help='Maximum number of races to archive')
    parser.add_argument('--clean', action='store_true', help='Clean up old archived races')
    parser.add_argument('--clean-days', type=int, default=30, help='Remove races older than this many days')
    parser.add_argument('--delete', action='store_true', help='Actually delete old races (otherwise dry run)')
    parser.add_argument('--force', action='store_true', help='Include already archived races')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Create archiver
    archiver = RaceArchiver(db_name=args.db, verbose=args.verbose)

    if args.clean and not args.from_date and not args.to_date and not args.limit:
        # Just run cleanup
        results = archiver.clean_archived_races(args.clean_days, not args.delete)

        print(f"\nCleanup results ({results['mode']}):")
        print(f"  Cutoff date: {results['cutoff_date']}")
        print(f"  Races to clean: {results['count']}")

        if args.verbose and results['races']:
            print("\nRaces that would be removed:")
            for race in results['races'][:10]:  # Show only first 10
                print(f"  {race['jour']} - {race['hippo']} - {race['prix']} ({race['comp']})")

            if len(results['races']) > 10:
                print(f"  ... and {len(results['races']) - 10} more")
    else:
        # Run full archiving pipeline
        results = archiver.run_archive_pipeline(
            date_from=args.from_date,
            date_to=args.to_date,
            limit=args.limit,
            clean=args.clean,
            clean_days=args.clean_days,
            dry_run=not args.delete
        )

        if results['status'] == 'success':
            archive_results = results['archive_results']

            print("\nArchiving results:")
            print(f"  Total races processed: {archive_results['total']}")
            print(f"  Successfully archived: {archive_results['successful']}")
            print(f"  Failed: {archive_results['failed']}")
            print(f"  Already archived: {archive_results['already_archived']}")

            if args.clean:
                cleanup_results = results['cleanup_results']
                print("\nCleanup results:")
                print(f"  Mode: {cleanup_results['mode']}")
                print(f"  Races to clean: {cleanup_results['count']}")
        else:
            print(f"Error: {results.get('message', 'Unknown error')}")

        print(f"\nExecution time: {results['execution_time']:.2f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())