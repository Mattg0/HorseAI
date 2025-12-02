"""
Batch Job Manager for Streamlit

Manages batch prediction jobs - launching, monitoring, and retrieving status.
Designed for use with Streamlit UI.
"""

import subprocess
import sqlite3
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class JobStatus:
    """Job status information."""
    job_id: str
    status: str
    progress: int
    message: str
    total_races: int
    processed_races: int
    successful_races: int
    failed_races: int
    start_time: Optional[str]
    end_time: Optional[str]
    error: Optional[str]
    config: Dict[str, Any]

    @property
    def is_running(self) -> bool:
        return self.status == 'running'

    @property
    def is_completed(self) -> bool:
        return self.status == 'completed'

    @property
    def is_failed(self) -> bool:
        return self.status == 'failed'

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if not self.start_time:
            return None

        start = datetime.fromisoformat(self.start_time)

        if self.end_time:
            end = datetime.fromisoformat(self.end_time)
        else:
            end = datetime.now()

        return (end - start).total_seconds()


class BatchJobManager:
    """Manages batch prediction jobs."""

    def __init__(self, db_path: str):
        """
        Initialize job manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Ensure prediction_jobs table exists."""
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

    def launch_job(
        self,
        date: str = None,
        race_ids: List[str] = None,
        limit: int = None,
        force_reprediction: bool = False,
        workers: int = -1,
        chunk_size: int = 50,
        max_memory_mb: float = 4096,
        db_name: str = None
    ) -> str:
        """
        Launch a batch prediction job as a background process.

        Args:
            date: Predict races from specific date (YYYY-MM-DD)
            race_ids: List of specific race IDs to predict
            limit: Maximum number of races to predict
            force_reprediction: Re-predict all races
            workers: Number of parallel workers (-1 = all cores)
            chunk_size: Races per chunk for memory management
            max_memory_mb: Maximum memory usage in MB
            db_name: Database name

        Returns:
            job_id: Unique job identifier
        """
        # Generate job ID
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Build command
        script_path = Path(__file__).parent.parent / "scripts" / "batch_predict.py"
        cmd = [sys.executable, str(script_path), "--job-id", job_id]

        if date:
            cmd.extend(["--date", date])

        if race_ids:
            cmd.extend(["--race-ids"] + race_ids)

        if limit:
            cmd.extend(["--limit", str(limit)])

        if force_reprediction:
            cmd.append("--force-reprediction")

        cmd.extend(["--workers", str(workers)])
        cmd.extend(["--chunk-size", str(chunk_size)])
        cmd.extend(["--max-memory", str(max_memory_mb)])

        if db_name:
            cmd.extend(["--db-name", db_name])

        # Launch process in background
        # Redirect output to log file
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{job_id}.log"

        with open(log_file, 'w') as f:
            subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True  # Detach from parent process
            )

        return job_id

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get status of a specific job.

        Args:
            job_id: Job identifier

        Returns:
            JobStatus or None if job not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT job_id, status, progress, message, total_races,
                   processed_races, successful_races, failed_races,
                   start_time, end_time, error, config
            FROM prediction_jobs
            WHERE job_id = ?
        """, (job_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        config = json.loads(row[11]) if row[11] else {}

        return JobStatus(
            job_id=row[0],
            status=row[1],
            progress=row[2],
            message=row[3] or "",
            total_races=row[4] or 0,
            processed_races=row[5] or 0,
            successful_races=row[6] or 0,
            failed_races=row[7] or 0,
            start_time=row[8],
            end_time=row[9],
            error=row[10],
            config=config
        )

    def list_jobs(
        self,
        status: str = None,
        limit: int = 10
    ) -> List[JobStatus]:
        """
        List jobs, optionally filtered by status.

        Args:
            status: Filter by status ('running', 'completed', 'failed')
            limit: Maximum number of jobs to return

        Returns:
            List of JobStatus objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if status:
            query = """
                SELECT job_id, status, progress, message, total_races,
                       processed_races, successful_races, failed_races,
                       start_time, end_time, error, config
                FROM prediction_jobs
                WHERE status = ?
                ORDER BY start_time DESC
                LIMIT ?
            """
            cursor.execute(query, (status, limit))
        else:
            query = """
                SELECT job_id, status, progress, message, total_races,
                       processed_races, successful_races, failed_races,
                       start_time, end_time, error, config
                FROM prediction_jobs
                ORDER BY start_time DESC
                LIMIT ?
            """
            cursor.execute(query, (limit,))

        rows = cursor.fetchall()
        conn.close()

        jobs = []
        for row in rows:
            config = json.loads(row[11]) if row[11] else {}
            jobs.append(JobStatus(
                job_id=row[0],
                status=row[1],
                progress=row[2],
                message=row[3] or "",
                total_races=row[4] or 0,
                processed_races=row[5] or 0,
                successful_races=row[6] or 0,
                failed_races=row[7] or 0,
                start_time=row[8],
                end_time=row[9],
                error=row[10],
                config=config
            ))

        return jobs

    def get_active_jobs(self) -> List[JobStatus]:
        """Get all currently running jobs."""
        return self.list_jobs(status='running', limit=100)

    def get_recent_jobs(self, limit: int = 10) -> List[JobStatus]:
        """Get recent jobs (any status)."""
        return self.list_jobs(limit=limit)

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from history.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM prediction_jobs WHERE job_id = ?", (job_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        # Also delete log file if exists
        if deleted:
            log_file = Path(__file__).parent.parent / "logs" / f"{job_id}.log"
            if log_file.exists():
                log_file.unlink()

        return deleted

    def get_job_log(self, job_id: str) -> Optional[str]:
        """
        Get log output for a job.

        Args:
            job_id: Job identifier

        Returns:
            Log content or None if not found
        """
        log_file = Path(__file__).parent.parent / "logs" / f"{job_id}.log"

        if not log_file.exists():
            return None

        try:
            with open(log_file, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading log: {e}"

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Delete completed/failed jobs older than specified days.

        Args:
            days: Delete jobs older than this many days

        Returns:
            Number of jobs deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = datetime.now().timestamp() - (days * 86400)

        cursor.execute("""
            SELECT job_id FROM prediction_jobs
            WHERE status IN ('completed', 'failed')
            AND datetime(start_time) < datetime(?, 'unixepoch')
        """, (cutoff,))

        old_jobs = [row[0] for row in cursor.fetchall()]

        for job_id in old_jobs:
            self.delete_job(job_id)

        conn.close()

        return len(old_jobs)
