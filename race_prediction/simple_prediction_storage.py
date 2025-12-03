"""
Simple Prediction Storage for Competitive Weighting Analysis

Streamlined storage system focused on capturing essential prediction data
for competitive weighting analysis and bias refinement.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging


class SimplePredictionStorage:
    """
    Simple, efficient prediction storage for competitive weighting analysis.
    Captures base predictions, weighting decisions, and results for systematic improvement.
    """

    def __init__(self, db_path: str = "data/prediction_analysis.db", verbose: bool = False):
        """
        Initialize the simple prediction storage system.

        Args:
            db_path: Path to SQLite database
            verbose: Enable detailed logging
        """
        self.db_path = Path(db_path)
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize the prediction storage database."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            cursor = conn.cursor()

            # Create main prediction results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS race_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT NOT NULL,
                    horse_id INTEGER NOT NULL,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Base Model Predictions
                    rf_prediction REAL,
                    tabnet_prediction REAL,

                    -- Ensemble Weighting
                    ensemble_weight_rf REAL,        -- RF weight (e.g., 0.40)
                    ensemble_weight_tabnet REAL,    -- TabNet weight (e.g., 0.60)
                    ensemble_prediction REAL,       -- Weighted average

                    -- Competitive Weighting Information
                    competitive_adjustment REAL,    -- Position adjustment applied
                    primary_advantage_type TEXT,     -- 'speed', 'track', 'class', 'form', 'none'
                    advantage_strength REAL,        -- Magnitude of competitive edge

                    -- Final Result
                    final_prediction REAL,

                    -- Post-Race Data (populated later)
                    actual_result INTEGER NULL,

                    -- Constraints
                    UNIQUE(race_id, horse_id)
                )
            """)

            # Create indexes for efficient analysis
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_race_horse ON race_predictions (race_id, horse_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_date ON race_predictions (prediction_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis ON race_predictions (primary_advantage_type, actual_result)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_race_analysis ON race_predictions (race_id, actual_result)")

            conn.commit()

        if self.verbose:
            self.logger.info(f"Simple prediction storage initialized at {self.db_path}")

    def store_race_predictions(self,
                             race_id: str,
                             predictions_data: List[Dict[str, Any]]) -> int:
        """
        Store predictions for a complete race.

        Args:
            race_id: Unique race identifier
            predictions_data: List of prediction dictionaries for each horse
                Each dict should contain:
                - horse_id: Horse identifier
                - rf_prediction: RF model prediction
                - tabnet_prediction: TabNet model prediction (optional)
                - ensemble_weight_rf: RF weight in ensemble
                - ensemble_weight_tabnet: TabNet weight in ensemble
                - ensemble_prediction: Weighted ensemble prediction
                - competitive_adjustment: Competitive adjustment applied
                - primary_advantage_type: Main competitive advantage
                - advantage_strength: Strength of competitive edge
                - final_prediction: Final adjusted prediction

        Returns:
            Number of records inserted
        """
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            inserted_count = 0
            current_time = datetime.now(timezone.utc).isoformat()

            for horse_data in predictions_data:
                # Retry logic for database locks
                max_retries = 3
                retry_count = 0

                while retry_count < max_retries:
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO race_predictions (
                                race_id, horse_id, prediction_date,
                                raw_rf_prediction, raw_tabnet_prediction,
                                rf_prediction, tabnet_prediction,
                                ensemble_weight_rf, ensemble_weight_tabnet, ensemble_prediction,
                                competitive_adjustment, primary_advantage_type, advantage_strength,
                                final_prediction
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            race_id,
                            horse_data.get('horse_id'),
                            current_time,
                            horse_data.get('raw_rf_prediction'),  # NEW: Raw predictions
                            horse_data.get('raw_tabnet_prediction'),  # NEW: Raw predictions
                            horse_data.get('rf_prediction'),  # Calibrated
                            horse_data.get('tabnet_prediction'),  # Calibrated
                            horse_data.get('ensemble_weight_rf'),
                            horse_data.get('ensemble_weight_tabnet'),
                            horse_data.get('ensemble_prediction'),
                            horse_data.get('competitive_adjustment', 0.0),
                            horse_data.get('primary_advantage_type', 'none'),
                            horse_data.get('advantage_strength', 0.0),
                            horse_data.get('final_prediction')
                        ))
                        inserted_count += 1
                        break  # Success, exit retry loop

                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e) and retry_count < max_retries - 1:
                            retry_count += 1
                            import time
                            time.sleep(0.1 * retry_count)  # Exponential backoff
                            if self.verbose:
                                self.logger.warning(f"Database locked, retrying {retry_count}/{max_retries} for horse {horse_data.get('horse_id')}")
                        else:
                            if self.verbose:
                                self.logger.error(f"Failed to insert prediction for horse {horse_data.get('horse_id')}: {e}")
                            break
                    except Exception as e:
                        if self.verbose:
                            self.logger.error(f"Failed to insert prediction for horse {horse_data.get('horse_id')}: {e}")
                        break

            conn.commit()

        if self.verbose:
            self.logger.info(f"Stored {inserted_count} predictions for race {race_id}")

        return inserted_count

    def update_race_results(self,
                          race_id: str,
                          results: Dict[int, int]) -> int:
        """
        Update actual results for a race.

        Args:
            race_id: Race identifier
            results: Dictionary mapping horse_id to final position

        Returns:
            Number of records updated
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            updated_count = 0

            for horse_id, position in results.items():
                cursor.execute("""
                    UPDATE race_predictions
                    SET actual_result = ?
                    WHERE race_id = ? AND horse_id = ?
                """, (position, race_id, horse_id))

                if cursor.rowcount > 0:
                    updated_count += 1

            conn.commit()

        if self.verbose:
            self.logger.info(f"Updated results for {updated_count} horses in race {race_id}")

        return updated_count

    def get_model_performance_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get model performance comparison over recent period.

        Args:
            days_back: Number of days to analyze

        Returns:
            Dictionary with performance metrics
        """
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_back)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN actual_result IS NOT NULL THEN 1 END) as validated_predictions,
                    AVG(CASE WHEN actual_result IS NOT NULL THEN ABS(rf_prediction - actual_result) END) as rf_mae,
                    AVG(CASE WHEN actual_result IS NOT NULL THEN ABS(tabnet_prediction - actual_result) END) as tabnet_mae,
                    AVG(CASE WHEN actual_result IS NOT NULL THEN ABS(ensemble_prediction - actual_result) END) as ensemble_mae,
                    AVG(CASE WHEN actual_result IS NOT NULL THEN ABS(final_prediction - actual_result) END) as final_mae,
                    AVG(CASE WHEN actual_result IS NOT NULL THEN competitive_adjustment END) as avg_competitive_adjustment,
                    0 as std_competitive_adjustment
                FROM race_predictions
                WHERE prediction_date > ?
            """

            result = pd.read_sql_query(query, conn, params=(cutoff_date,))

        if len(result) == 0:
            return {}

        row = result.iloc[0]

        return {
            'analysis_period_days': days_back,
            'total_predictions': int(row['total_predictions']),
            'validated_predictions': int(row['validated_predictions']),
            'validation_rate': row['validated_predictions'] / row['total_predictions'] if row['total_predictions'] > 0 else 0,
            'model_performance': {
                'rf_mae': float(row['rf_mae']) if pd.notna(row['rf_mae']) else None,
                'tabnet_mae': float(row['tabnet_mae']) if pd.notna(row['tabnet_mae']) else None,
                'ensemble_mae': float(row['ensemble_mae']) if pd.notna(row['ensemble_mae']) else None,
                'final_mae': float(row['final_mae']) if pd.notna(row['final_mae']) else None
            },
            'competitive_weighting': {
                'avg_adjustment': float(row['avg_competitive_adjustment']) if pd.notna(row['avg_competitive_adjustment']) else 0,
                'std_adjustment': float(row['std_competitive_adjustment']) if pd.notna(row['std_competitive_adjustment']) else 0
            }
        }

    def get_competitive_weighting_analysis(self, days_back: int = 30) -> pd.DataFrame:
        """
        Analyze competitive weighting effectiveness by advantage type.

        Args:
            days_back: Number of days to analyze

        Returns:
            DataFrame with competitive weighting analysis
        """
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_back)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT
                    primary_advantage_type,
                    COUNT(*) as predictions,
                    COUNT(CASE WHEN actual_result IS NOT NULL THEN 1 END) as validated,
                    AVG(competitive_adjustment) as avg_adjustment,
                    0 as std_adjustment,
                    AVG(advantage_strength) as avg_advantage_strength,
                    AVG(CASE WHEN actual_result IS NOT NULL THEN ABS(ensemble_prediction - actual_result) END) as ensemble_mae,
                    AVG(CASE WHEN actual_result IS NOT NULL THEN ABS(final_prediction - actual_result) END) as final_mae,
                    AVG(CASE WHEN actual_result IS NOT NULL THEN
                        ABS(final_prediction - actual_result) - ABS(ensemble_prediction - actual_result)
                    END) as mae_improvement,
                    COUNT(CASE WHEN actual_result IS NOT NULL AND actual_result <= 3 THEN 1 END) as top3_actual,
                    COUNT(CASE WHEN actual_result IS NOT NULL AND final_prediction <= 3 THEN 1 END) as top3_predicted
                FROM race_predictions
                WHERE prediction_date > ?
                GROUP BY primary_advantage_type
                ORDER BY predictions DESC
            """

            df = pd.read_sql_query(query, conn, params=(cutoff_date,))

        # Calculate additional metrics
        if len(df) > 0:
            df['validation_rate'] = df['validated'] / df['predictions']
            df['top3_accuracy'] = df['top3_actual'] / df['validated'].clip(lower=1)
            df['top3_precision'] = df['top3_actual'] / df['top3_predicted'].clip(lower=1)

        return df

    def get_recent_predictions(self, limit: int = 100) -> pd.DataFrame:
        """
        Get recent predictions for review.

        Args:
            limit: Maximum number of records to return

        Returns:
            DataFrame with recent predictions
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT
                    race_id,
                    horse_id,
                    prediction_date,
                    rf_prediction,
                    tabnet_prediction,
                    ensemble_prediction,
                    competitive_adjustment,
                    primary_advantage_type,
                    advantage_strength,
                    final_prediction,
                    actual_result,
                    CASE WHEN actual_result IS NOT NULL THEN
                        ABS(final_prediction - actual_result)
                    END as prediction_error
                FROM race_predictions
                ORDER BY prediction_date DESC
                LIMIT ?
            """

            df = pd.read_sql_query(query, conn, params=(limit,))

        return df

    def get_race_prediction_summary(self, race_id: str) -> Dict[str, Any]:
        """
        Get complete prediction summary for a specific race.

        Args:
            race_id: Race identifier

        Returns:
            Dictionary with race prediction summary
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get race predictions
            query = """
                SELECT * FROM race_predictions
                WHERE race_id = ?
                ORDER BY final_prediction
            """

            df = pd.read_sql_query(query, conn, params=(race_id,))

        if len(df) == 0:
            return {}

        # Calculate race-level statistics
        summary = {
            'race_id': race_id,
            'prediction_date': df['prediction_date'].iloc[0],
            'total_horses': len(df),
            'horses_with_results': len(df[df['actual_result'].notna()]),
            'predictions': df.to_dict('records'),
            'statistics': {}
        }

        # Add statistics if we have actual results
        if summary['horses_with_results'] > 0:
            validated_df = df[df['actual_result'].notna()]

            summary['statistics'] = {
                'avg_final_mae': float(validated_df['actual_result'].sub(validated_df['final_prediction']).abs().mean()),
                'avg_ensemble_mae': float(validated_df['actual_result'].sub(validated_df['ensemble_prediction']).abs().mean()),
                'avg_competitive_adjustment': float(df['competitive_adjustment'].mean()),
                'advantage_types': df['primary_advantage_type'].value_counts().to_dict(),
                'top3_predicted_correct': len(validated_df[(validated_df['final_prediction'] <= 3) & (validated_df['actual_result'] <= 3)]),
                'top3_actual': len(validated_df[validated_df['actual_result'] <= 3])
            }

        return summary

    def export_analysis_data(self, output_path: str, days_back: int = 30) -> str:
        """
        Export prediction data for external analysis.

        Args:
            output_path: Path for output CSV file
            days_back: Number of days to include

        Returns:
            Path to exported file
        """
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_back)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM race_predictions
                WHERE prediction_date > ?
                ORDER BY prediction_date DESC, race_id, final_prediction
            """

            df = pd.read_sql_query(query, conn, params=(cutoff_date,))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        if self.verbose:
            self.logger.info(f"Exported {len(df)} prediction records to {output_path}")

        return str(output_path)

    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old prediction data to manage database size.

        Args:
            days_to_keep: Number of days of data to retain

        Returns:
            Number of records deleted
        """
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_to_keep)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM race_predictions WHERE prediction_date < ?", (cutoff_date,))
            count_to_delete = cursor.fetchone()[0]

            if count_to_delete > 0:
                cursor.execute("DELETE FROM race_predictions WHERE prediction_date < ?", (cutoff_date,))
                conn.commit()

                if self.verbose:
                    self.logger.info(f"Cleaned up {count_to_delete} old prediction records")

            return count_to_delete

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total records
            cursor.execute("SELECT COUNT(*) FROM race_predictions")
            total_records = cursor.fetchone()[0]

            # Records with results
            cursor.execute("SELECT COUNT(*) FROM race_predictions WHERE actual_result IS NOT NULL")
            validated_records = cursor.fetchone()[0]

            # Date range
            cursor.execute("SELECT MIN(prediction_date), MAX(prediction_date) FROM race_predictions")
            min_date, max_date = cursor.fetchone()

            # Unique races
            cursor.execute("SELECT COUNT(DISTINCT race_id) FROM race_predictions")
            unique_races = cursor.fetchone()[0]

        return {
            'total_records': total_records,
            'validated_records': validated_records,
            'validation_rate': validated_records / total_records if total_records > 0 else 0,
            'unique_races': unique_races,
            'date_range': {
                'earliest': min_date,
                'latest': max_date
            },
            'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }


# Helper function to extract prediction data from competitive analysis results
def extract_prediction_data_from_competitive_analysis(race_id: str,
                                                    race_data: pd.DataFrame,
                                                    base_predictions: Dict[str, np.ndarray],
                                                    competitive_results: Dict[str, Any],
                                                    final_predictions: np.ndarray,
                                                    blend_weights: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Extract prediction data from competitive analysis results for storage.

    Args:
        race_id: Race identifier
        race_data: Original race data DataFrame
        base_predictions: Base model predictions
        competitive_results: Results from competitive analysis
        final_predictions: Final blended predictions
        blend_weights: Model blending weights

    Returns:
        List of prediction data dictionaries ready for storage
    """
    predictions_data = []

    # Extract competitive analysis details
    competitive_scores = competitive_results.get('competitive_analysis', {}).get('competitive_scores', {})
    adjustment_details = competitive_results.get('audit_trail', {}).get('adjustment_summary', {})

    for i, (idx, row) in enumerate(race_data.iterrows()):
        # Use actual horse ID (idche) instead of race number (numero)
        horse_id = row.get('idche', row.get('numero', idx))

        # Get base predictions with better debugging
        rf_pred = None
        tabnet_pred = None

        if 'rf' in base_predictions and base_predictions['rf'] is not None:
            rf_pred = base_predictions['rf'][i] if i < len(base_predictions['rf']) else None

        if 'tabnet' in base_predictions and base_predictions['tabnet'] is not None:
            tabnet_pred = base_predictions['tabnet'][i] if i < len(base_predictions['tabnet']) else None

        # Debug TabNet values (only for first horse)
        if i == 0 and 'tabnet' in base_predictions and base_predictions['tabnet'] is not None:
            print(f"Storage: TabNet predictions available ({len(base_predictions['tabnet'])} horses)")

        # Calculate ensemble prediction
        ensemble_pred = None
        if rf_pred is not None and tabnet_pred is not None:
            rf_weight = blend_weights.get('rf', 0.5)
            tabnet_weight = blend_weights.get('tabnet', 0.5)
            ensemble_pred = (rf_pred * rf_weight) + (tabnet_pred * tabnet_weight)
        elif rf_pred is not None:
            ensemble_pred = rf_pred
            rf_weight = 1.0
            tabnet_weight = 0.0
        elif tabnet_pred is not None:
            ensemble_pred = tabnet_pred
            rf_weight = 0.0
            tabnet_weight = 1.0
        else:
            rf_weight = tabnet_weight = 0.0

        # CRITICAL FIX: Get competitive analysis data using DataFrame row index (not horse_id)
        # The competitive_scores is indexed by DataFrame row index (i), not horse_id
        horse_competitive = competitive_scores.get(i, {})
        competitive_adjustment = final_predictions[i] - ensemble_pred if ensemble_pred is not None else 0.0

        # Determine primary advantage type
        advantages = horse_competitive.get('advantages', {})
        primary_advantage = 'none'
        advantage_strength = 0.0

        if advantages:
            # Find the strongest advantage
            advantage_scores = {
                'speed': horse_competitive.get('speed_score', 0),
                'track': horse_competitive.get('track_score', 0),
                'class': horse_competitive.get('class_score', 0),
                'form': horse_competitive.get('form_score', 0)
            }

            # Get the advantage with highest score
            if advantage_scores:
                primary_advantage = max(advantage_scores.items(), key=lambda x: abs(x[1]))[0]
                advantage_strength = advantage_scores[primary_advantage]

        prediction_data = {
            'horse_id': int(horse_id),
            'rf_prediction': float(rf_pred) if rf_pred is not None else None,
            'tabnet_prediction': float(tabnet_pred) if tabnet_pred is not None else None,
            'ensemble_weight_rf': float(rf_weight),
            'ensemble_weight_tabnet': float(tabnet_weight),
            'ensemble_prediction': float(ensemble_pred) if ensemble_pred is not None else None,
            'competitive_adjustment': float(competitive_adjustment),
            'primary_advantage_type': primary_advantage,
            'advantage_strength': float(advantage_strength),
            'final_prediction': float(final_predictions[i])
        }

        predictions_data.append(prediction_data)

    return predictions_data