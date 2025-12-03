#!/usr/bin/env python3
"""
Batch Prediction CLI Script

Runs batch predictions in the background with database progress tracking.
Designed to be launched from Streamlit or run independently from command line.

Usage:
    # Predict all unpredicted races
    python scripts/batch_predict.py

    # Predict specific date
    python scripts/batch_predict.py --date 2025-10-15

    # Predict with custom settings
    python scripts/batch_predict.py --limit 1000 --workers 8 --chunk-size 30

    # Force reprediction of all races
    python scripts/batch_predict.py --force-reprediction
"""

import argparse
import sys
import os
import time
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from race_prediction.race_predict import predict_races_fast
from utils.env_setup import AppConfig


class BatchPredictionJob:
    """Manages a batch prediction job with progress tracking."""

    def __init__(self, job_id: str, db_path: str):
        """
        Initialize job tracker.

        Args:
            job_id: Unique job identifier
            db_path: Path to SQLite database
        """
        self.job_id = job_id
        self.db_path = db_path
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create prediction_jobs table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER DEFAULT 0,
                message TEXT,
                total_races INTEGER,
                processed_races INTEGER DEFAULT 0,
                successful_races INTEGER DEFAULT 0,
                failed_races INTEGER DEFAULT 0,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                error TEXT,
                config TEXT
            )
        """)

        conn.commit()
        conn.close()

    def start(self, total_races: int, config: dict):
        """Mark job as started."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO prediction_jobs
            (job_id, status, progress, message, total_races, start_time, config)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            self.job_id,
            'running',
            0,
            'Starting batch prediction...',
            total_races,
            datetime.now().isoformat(),
            json.dumps(config)
        ))

        conn.commit()
        conn.close()

    def update_progress(self, progress: int, message: str,
                       processed: int = None, successful: int = None):
        """Update job progress."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updates = {
            'progress': progress,
            'message': message
        }

        if processed is not None:
            updates['processed_races'] = processed
        if successful is not None:
            updates['successful_races'] = successful

        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [self.job_id]

        cursor.execute(f"""
            UPDATE prediction_jobs
            SET {set_clause}
            WHERE job_id = ?
        """, values)

        conn.commit()
        conn.close()

    def complete(self, successful: int, failed: int, message: str):
        """Mark job as completed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE prediction_jobs
            SET status = ?, progress = 100, message = ?,
                successful_races = ?, failed_races = ?,
                end_time = ?
            WHERE job_id = ?
        """, ('completed', message, successful, failed,
              datetime.now().isoformat(), self.job_id))

        conn.commit()
        conn.close()

    def fail(self, error: str):
        """Mark job as failed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE prediction_jobs
            SET status = ?, message = ?, error = ?,
                end_time = ?
            WHERE job_id = ?
        """, ('failed', f'Job failed: {error}', error,
              datetime.now().isoformat(), self.job_id))

        conn.commit()
        conn.close()


def get_races_to_predict(db_path: str, date: str = None,
                         force_reprediction: bool = False,
                         limit: int = None) -> List[str]:
    """Get list of race IDs to predict."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if force_reprediction:
        # All races that have been processed (have participants)
        if date:
            query = """
                SELECT comp FROM daily_race
                WHERE participants IS NOT NULL
                AND participants != '[]'
                AND jour = ?
            """
            params = (date,)
        else:
            query = """
                SELECT comp FROM daily_race
                WHERE participants IS NOT NULL
                AND participants != '[]'
            """
            params = ()
    else:
        # Only unpredicted races
        if date:
            query = """
                SELECT comp FROM daily_race
                WHERE (prediction_results IS NULL OR prediction_results = '')
                AND participants IS NOT NULL
                AND participants != '[]'
                AND jour = ?
            """
            params = (date,)
        else:
            query = """
                SELECT comp FROM daily_race
                WHERE (prediction_results IS NULL OR prediction_results = '')
                AND participants IS NOT NULL
                AND participants != '[]'
            """
            params = ()

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, params)
    race_ids = [row[0] for row in cursor.fetchall()]

    conn.close()
    return race_ids


def run_batch_prediction(
    job_id: str = None,
    date: str = None,
    race_ids: List[str] = None,
    limit: int = None,
    force_reprediction: bool = False,
    n_jobs: int = -1,
    chunk_size: int = 50,
    max_memory_mb: float = 4096,
    db_name: str = None
):
    """
    Run batch prediction with progress tracking.

    Args:
        job_id: Unique job identifier (auto-generated if None)
        date: Predict races from specific date (YYYY-MM-DD)
        race_ids: List of specific race IDs to predict
        limit: Maximum number of races to predict
        force_reprediction: Re-predict all races (not just unpredicted)
        n_jobs: Number of parallel workers (-1 = all cores)
        chunk_size: Races per chunk for memory management
        max_memory_mb: Maximum memory usage in MB
        db_name: Database name (defaults to active_db)
    """
    # Initialize config and paths
    config = AppConfig()
    if db_name is None:
        db_name = config._config.base.active_db
    db_path = config.get_sqlite_dbpath(db_name)

    # Generate job ID if not provided
    if job_id is None:
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize job tracker
    job = BatchPredictionJob(job_id, db_path)

    try:
        # Get races to predict
        if race_ids is None:
            race_ids = get_races_to_predict(
                db_path,
                date=date,
                force_reprediction=force_reprediction,
                limit=limit
            )

        total_races = len(race_ids)

        if total_races == 0:
            job.complete(0, 0, "No races to predict")
            print("No races to predict")
            return

        # Start job
        job_config = {
            'date': date,
            'limit': limit,
            'force_reprediction': force_reprediction,
            'n_jobs': n_jobs,
            'chunk_size': chunk_size,
            'max_memory_mb': max_memory_mb,
            'total_races': total_races
        }
        job.start(total_races, job_config)

        print(f"🚀 Starting batch prediction job: {job_id}")
        print(f"   Total races: {total_races}")
        print(f"   Workers: {n_jobs if n_jobs > 0 else 'all CPU cores'}")
        print(f"   Chunk size: {chunk_size}")
        print(f"   Memory limit: {max_memory_mb}MB")
        print(f"   Database: {db_name}")
        print()

        # Progress callback
        def progress_callback(percent: int, message: str):
            # Calculate approximate races processed
            processed = int((percent / 100.0) * total_races)
            job.update_progress(percent, message, processed=processed)
            print(f"[{percent}%] {message}")

        # Run batch prediction
        start_time = time.time()

        results = predict_races_fast(
            race_ids=race_ids,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            max_memory_mb=max_memory_mb,
            db_name=db_name,
            verbose=False,
            progress_callback=progress_callback
        )

        total_time = time.time() - start_time

        # Count successful predictions
        successful = len([r for r in results if r is not None and len(r) > 0])
        failed = total_races - successful

        # Complete job
        message = f"Completed {successful}/{total_races} races in {total_time:.1f}s ({total_races/total_time:.1f} races/sec)"
        job.complete(successful, failed, message)

        print()
        print("=" * 60)
        print("✅ BATCH PREDICTION COMPLETE")
        print("=" * 60)
        print(f"  Job ID: {job_id}")
        print(f"  Total races: {total_races}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {total_races/total_time:.1f} races/second")
        print("=" * 60)

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        job.fail(error_msg)
        print(f"❌ ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Batch race prediction with progress tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict all unpredicted races
  python scripts/batch_predict.py

  # Predict specific date
  python scripts/batch_predict.py --date 2025-10-15

  # Predict 1000 races with 8 workers
  python scripts/batch_predict.py --limit 1000 --workers 8

  # Force reprediction of all races
  python scripts/batch_predict.py --force-reprediction

  # Custom memory settings
  python scripts/batch_predict.py --chunk-size 30 --max-memory 2048
        """
    )

    parser.add_argument(
        '--job-id',
        type=str,
        help='Unique job identifier (auto-generated if not provided)'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Predict races from specific date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--race-ids',
        type=str,
        nargs='+',
        help='Specific race IDs to predict'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of races to predict'
    )

    parser.add_argument(
        '--force-reprediction',
        action='store_true',
        help='Re-predict all races (not just unpredicted ones)'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=-1,
        help='Number of parallel workers (-1 = all CPU cores, default: -1)'
    )

    parser.add_argument(
        '--chunk-size', '-c',
        type=int,
        default=50,
        help='Races per chunk for memory management (default: 50)'
    )

    parser.add_argument(
        '--max-memory', '-m',
        type=float,
        default=4096,
        help='Maximum memory usage in MB (default: 4096)'
    )

    parser.add_argument(
        '--db-name',
        type=str,
        help='Database name (defaults to active_db from config)'
    )

    args = parser.parse_args()

    # Run batch prediction
    run_batch_prediction(
        job_id=args.job_id,
        date=args.date,
        race_ids=args.race_ids,
        limit=args.limit,
        force_reprediction=args.force_reprediction,
        n_jobs=args.workers,
        chunk_size=args.chunk_size,
        max_memory_mb=args.max_memory,
        db_name=args.db_name
    )


if __name__ == "__main__":
    main()
