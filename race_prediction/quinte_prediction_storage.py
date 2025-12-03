"""
Quinté Prediction Storage

Dedicated storage system for Quinté+ race predictions with Quinté-specific fields.
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging


class QuintePredictionStorage:
    """
    Dedicated prediction storage for Quinté+ races.

    Stores Quinté predictions separately from general race predictions,
    with additional Quinté-specific fields for analysis.
    """

    def __init__(self, db_path: str = "data/hippique2.db", verbose: bool = False):
        """
        Initialize Quinté prediction storage.

        Args:
            db_path: Path to SQLite database
            verbose: Enable detailed logging
        """
        self.db_path = Path(db_path)
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize the Quinté prediction storage table."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            cursor = conn.cursor()

            # Create Quinté predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quinte_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT NOT NULL,
                    horse_id INTEGER NOT NULL,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Quinté-specific metadata
                    race_date TEXT,              -- Date of the race (YYYY-MM-DD)
                    track TEXT,                  -- Hippodrome
                    race_number INTEGER,         -- Prix number
                    race_name TEXT,              -- Prize name

                    -- Model Predictions (Quinté models)
                    quinte_rf_prediction REAL,           -- Quinté RF model prediction
                    quinte_tabnet_prediction REAL,       -- Quinté TabNet model prediction

                    -- General Model Predictions (for blending)
                    general_rf_prediction REAL,          -- General RF model prediction
                    general_tabnet_prediction REAL,      -- General TabNet model prediction

                    -- Ensemble Weighting
                    quinte_weight REAL,                  -- Weight given to Quinté model (e.g., 0.20)
                    general_weight REAL,                 -- Weight given to General model (e.g., 0.80)
                    ensemble_weight_rf REAL,             -- RF weight in ensemble
                    ensemble_weight_tabnet REAL,         -- TabNet weight in ensemble
                    ensemble_prediction REAL,            -- Weighted average prediction

                    -- Competitive Analysis
                    competitive_adjustment REAL,         -- Position adjustment from competitive analysis
                    primary_advantage_type TEXT,         -- Type of competitive advantage
                    advantage_strength REAL,             -- Strength of advantage

                    -- Calibration (Quinté-specific)
                    calibrated_rf_prediction REAL,       -- Calibrated RF prediction
                    calibrated_tabnet_prediction REAL,   -- Calibrated TabNet prediction
                    calibration_applied BOOLEAN DEFAULT 0, -- Whether calibration was applied

                    -- Final Result
                    final_prediction REAL,               -- Final position prediction
                    predicted_rank INTEGER,              -- Predicted rank (1 = winner)

                    -- Horse Information
                    horse_number INTEGER,                -- Numero
                    horse_name TEXT,                     -- Nom

                    -- Quinté-specific Features
                    is_favorite BOOLEAN DEFAULT 0,       -- Is this horse a favorite?
                    quinte_score REAL,                   -- Quinté-specific performance score
                    quinte_form_rating REAL,             -- Form rating for Quinté races

                    -- Post-Race Data (populated after race)
                    actual_result INTEGER NULL,          -- Actual finishing position
                    was_in_quinte BOOLEAN NULL,          -- Was in top 5 (Quinté)

                    -- Constraints
                    UNIQUE(race_id, horse_id)
                )
            """)

            # Create indexes for efficient querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quinte_race_horse ON quinte_predictions (race_id, horse_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quinte_prediction_date ON quinte_predictions (prediction_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quinte_race_date ON quinte_predictions (race_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quinte_analysis ON quinte_predictions (primary_advantage_type, actual_result)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quinte_performance ON quinte_predictions (race_id, predicted_rank, actual_result)")

            # Create summary table for race-level Quinté predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quinte_race_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT UNIQUE NOT NULL,
                    race_date TEXT,
                    track TEXT,
                    race_number INTEGER,
                    race_name TEXT,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Prediction Summary
                    predicted_quinte TEXT,               -- Top 5 predicted horses (comma-separated numbers)
                    predicted_winner INTEGER,            -- Predicted winner numero
                    total_horses INTEGER,                -- Number of horses in race

                    -- Model Performance (after results)
                    quinte_accuracy REAL NULL,           -- How many of top 5 were correct
                    winner_correct BOOLEAN NULL,         -- Did we predict the winner?

                    -- Metadata
                    quinte_weight REAL,                  -- Quinté model weight used
                    general_weight REAL,                 -- General model weight used
                    calibration_applied BOOLEAN DEFAULT 0
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quinte_race_summary_date ON quinte_race_summary (race_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quinte_race_summary_id ON quinte_race_summary (race_id)")

            conn.commit()

        if self.verbose:
            self.logger.info(f"Quinté prediction storage initialized at {self.db_path}")

    def store_quinte_predictions(
        self,
        race_id: str,
        predictions_data: List[Dict[str, Any]],
        race_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store Quinté predictions for a complete race.

        Args:
            race_id: Unique race identifier
            predictions_data: List of prediction dictionaries (one per horse)
            race_metadata: Optional race-level metadata

        Returns:
            Number of horses stored
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        cursor = conn.cursor()

        stored_count = 0

        try:
            for horse_data in predictions_data:
                try:
                    # Insert or replace prediction
                    cursor.execute("""
                        INSERT OR REPLACE INTO quinte_predictions (
                            race_id, horse_id, race_date, track, race_number, race_name,
                            quinte_rf_prediction, quinte_tabnet_prediction,
                            general_rf_prediction, general_tabnet_prediction,
                            quinte_weight, general_weight,
                            ensemble_weight_rf, ensemble_weight_tabnet, ensemble_prediction,
                            competitive_adjustment, primary_advantage_type, advantage_strength,
                            calibrated_rf_prediction, calibrated_tabnet_prediction, calibration_applied,
                            final_prediction, predicted_rank,
                            horse_number, horse_name,
                            is_favorite, quinte_score, quinte_form_rating
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        race_id,
                        horse_data.get('horse_id'),
                        race_metadata.get('race_date') if race_metadata else None,
                        race_metadata.get('track') if race_metadata else None,
                        race_metadata.get('race_number') if race_metadata else None,
                        race_metadata.get('race_name') if race_metadata else None,
                        horse_data.get('quinte_rf_prediction'),
                        horse_data.get('quinte_tabnet_prediction'),
                        horse_data.get('general_rf_prediction'),
                        horse_data.get('general_tabnet_prediction'),
                        horse_data.get('quinte_weight', 0.2),
                        horse_data.get('general_weight', 0.8),
                        horse_data.get('ensemble_weight_rf'),
                        horse_data.get('ensemble_weight_tabnet'),
                        horse_data.get('ensemble_prediction'),
                        horse_data.get('competitive_adjustment', 0.0),
                        horse_data.get('primary_advantage_type', 'none'),
                        horse_data.get('advantage_strength', 0.0),
                        horse_data.get('calibrated_rf_prediction'),
                        horse_data.get('calibrated_tabnet_prediction'),
                        horse_data.get('calibration_applied', False),
                        horse_data.get('final_prediction'),
                        horse_data.get('predicted_rank'),
                        horse_data.get('horse_number'),
                        horse_data.get('horse_name'),
                        horse_data.get('is_favorite', False),
                        horse_data.get('quinte_score'),
                        horse_data.get('quinte_form_rating')
                    ))

                    stored_count += 1

                except Exception as e:
                    if self.verbose:
                        self.logger.warning(f"Failed to store horse {horse_data.get('horse_id')}: {e}")

            # Store race summary if metadata provided
            if race_metadata and stored_count > 0:
                self._store_race_summary(cursor, race_id, race_metadata)

            conn.commit()

        finally:
            conn.close()

        if self.verbose:
            self.logger.info(f"Stored {stored_count} horses for Quinté race {race_id}")

        return stored_count

    def _store_race_summary(self, cursor, race_id: str, race_metadata: Dict[str, Any]):
        """Store race-level summary."""
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO quinte_race_summary (
                    race_id, race_date, track, race_number, race_name,
                    predicted_quinte, predicted_winner, total_horses,
                    quinte_weight, general_weight, calibration_applied
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id,
                race_metadata.get('race_date'),
                race_metadata.get('track'),
                race_metadata.get('race_number'),
                race_metadata.get('race_name'),
                race_metadata.get('predicted_quinte'),
                race_metadata.get('predicted_winner'),
                race_metadata.get('total_horses'),
                race_metadata.get('quinte_weight', 0.2),
                race_metadata.get('general_weight', 0.8),
                race_metadata.get('calibration_applied', False)
            ))
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Failed to store race summary for {race_id}: {e}")

    def update_actual_results(self, race_id: str, results: Dict[int, int]) -> int:
        """
        Update actual results after race completion.

        Args:
            race_id: Race identifier
            results: Dict mapping horse_id to finishing position

        Returns:
            Number of records updated
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        cursor = conn.cursor()

        updated = 0
        try:
            for horse_id, position in results.items():
                cursor.execute("""
                    UPDATE quinte_predictions
                    SET actual_result = ?,
                        was_in_quinte = ?
                    WHERE race_id = ? AND horse_id = ?
                """, (position, position <= 5, race_id, horse_id))

                if cursor.rowcount > 0:
                    updated += 1

            conn.commit()

        finally:
            conn.close()

        if self.verbose:
            self.logger.info(f"Updated {updated} results for race {race_id}")

        return updated

    def get_race_predictions(self, race_id: str) -> pd.DataFrame:
        """Get all predictions for a specific race."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)

        query = """
            SELECT * FROM quinte_predictions
            WHERE race_id = ?
            ORDER BY predicted_rank
        """

        df = pd.read_sql_query(query, conn, params=(race_id,))
        conn.close()

        return df

    def get_predictions_by_date(self, race_date: str) -> pd.DataFrame:
        """Get all Quinté predictions for a specific date."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)

        query = """
            SELECT * FROM quinte_predictions
            WHERE race_date = ?
            ORDER BY race_id, predicted_rank
        """

        df = pd.read_sql_query(query, conn, params=(race_date,))
        conn.close()

        return df

    def get_race_summary(self, race_id: str = None, race_date: str = None) -> pd.DataFrame:
        """Get race-level summaries."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)

        if race_id:
            query = "SELECT * FROM quinte_race_summary WHERE race_id = ?"
            df = pd.read_sql_query(query, conn, params=(race_id,))
        elif race_date:
            query = "SELECT * FROM quinte_race_summary WHERE race_date = ?"
            df = pd.read_sql_query(query, conn, params=(race_date,))
        else:
            query = "SELECT * FROM quinte_race_summary ORDER BY race_date DESC LIMIT 100"
            df = pd.read_sql_query(query, conn)

        conn.close()
        return df
