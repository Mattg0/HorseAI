import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass

from utils.env_setup import AppConfig


@dataclass
class PredictionRecord:
    race_id: str
    timestamp: datetime
    horse_id: str
    rf_prediction: Optional[float]
    lstm_prediction: Optional[float]
    tabnet_prediction: Optional[float]
    feedforward_prediction: Optional[float]
    transformer_prediction: Optional[float]
    ensemble_alt_prediction: Optional[float]
    ensemble_prediction: float
    ensemble_confidence_score: Optional[float]
    actual_position: Optional[int]
    distance: Optional[int]
    track_condition: Optional[str]
    weather: Optional[str]
    field_size: Optional[int]
    race_type: Optional[str]
    model_versions: Dict[str, str]
    prediction_metadata: Dict[str, Any]


class PredictionStorage:
    def __init__(self, config: AppConfig, db_path: Optional[str] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database configuration - use active_db instead of separate file
        if db_path:
            self.db_path = db_path
        else:
            # Use the active database from config for proper joins
            self.db_path = config.get_sqlite_dbpath(config._config.base.active_db)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with prediction storage schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS prediction_storage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    horse_id TEXT NOT NULL,
                    rf_prediction REAL,
                    lstm_prediction REAL,
                    tabnet_prediction REAL,
                    feedforward_prediction REAL,
                    transformer_prediction REAL,
                    ensemble_alt_prediction REAL,
                    ensemble_prediction REAL NOT NULL,
                    ensemble_confidence_score REAL,
                    actual_position INTEGER,
                    distance INTEGER,
                    track_condition TEXT,
                    weather TEXT,
                    field_size INTEGER,
                    race_type TEXT,
                    model_versions TEXT,  -- JSON string
                    prediction_metadata TEXT,  -- JSON string
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add new columns to existing tables (for backward compatibility)
            try:
                conn.execute('ALTER TABLE prediction_storage ADD COLUMN feedforward_prediction REAL')
                conn.execute('ALTER TABLE prediction_storage ADD COLUMN transformer_prediction REAL')
                conn.execute('ALTER TABLE prediction_storage ADD COLUMN ensemble_alt_prediction REAL')
            except sqlite3.OperationalError:
                # Columns already exist
                pass
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_race_id ON prediction_storage(race_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_storage(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_horse_id ON prediction_storage(horse_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_race_timestamp ON prediction_storage(race_id, timestamp)')
            
            conn.commit()
        
        self.logger.info(f"Initialized prediction storage database at {self.db_path}")
    
    def store_prediction(self, record: PredictionRecord) -> int:
        """Store individual race prediction"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO prediction_storage (
                    race_id, timestamp, horse_id, rf_prediction, lstm_prediction,
                    tabnet_prediction, feedforward_prediction, transformer_prediction,
                    ensemble_alt_prediction, ensemble_prediction, ensemble_confidence_score,
                    actual_position, distance, track_condition, weather, field_size,
                    race_type, model_versions, prediction_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.race_id,
                record.timestamp,
                record.horse_id,
                record.rf_prediction,
                record.lstm_prediction,
                record.tabnet_prediction,
                record.feedforward_prediction,
                record.transformer_prediction,
                record.ensemble_alt_prediction,
                record.ensemble_prediction,
                record.ensemble_confidence_score,
                record.actual_position,
                record.distance,
                record.track_condition,
                record.weather,
                record.field_size,
                record.race_type,
                json.dumps(record.model_versions),
                json.dumps(record.prediction_metadata)
            ))
            
            record_id = cursor.lastrowid
            conn.commit()
            
        self.logger.debug(f"Stored prediction record {record_id} for race {record.race_id}")
        return record_id
    
    def store_batch_predictions(self, records: List[PredictionRecord]) -> List[int]:
        """Store multiple race predictions at once"""
        record_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for record in records:
                cursor = conn.execute('''
                    INSERT INTO prediction_storage (
                        race_id, timestamp, horse_id, rf_prediction, lstm_prediction,
                        tabnet_prediction, feedforward_prediction, transformer_prediction,
                        ensemble_alt_prediction, ensemble_prediction, ensemble_confidence_score,
                        actual_position, distance, track_condition, weather, field_size,
                        race_type, model_versions, prediction_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.race_id,
                    record.timestamp,
                    record.horse_id,
                    record.rf_prediction,
                    record.lstm_prediction,
                    record.tabnet_prediction,
                    record.feedforward_prediction,
                    record.transformer_prediction,
                    record.ensemble_alt_prediction,
                    record.ensemble_prediction,
                    record.ensemble_confidence_score,
                    record.actual_position,
                    record.distance,
                    record.track_condition,
                    record.weather,
                    record.field_size,
                    record.race_type,
                    json.dumps(record.model_versions),
                    json.dumps(record.prediction_metadata)
                ))
                record_ids.append(cursor.lastrowid)
            
            conn.commit()
        
        self.logger.info(f"Stored {len(record_ids)} prediction records in batch")
        return record_ids
    
    def update_actual_results(self, race_id: str, results: Dict[str, int]):
        """Update actual race positions for a race"""
        with sqlite3.connect(self.db_path) as conn:
            for horse_id, position in results.items():
                conn.execute('''
                    UPDATE prediction_storage 
                    SET actual_position = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE race_id = ? AND horse_id = ?
                ''', (position, race_id, horse_id))
            
            conn.commit()
        
        self.logger.info(f"Updated actual results for race {race_id}")
    
    def get_recent_performance(self, days: int = 30) -> Dict[str, Any]:
        """Retrieve performance metrics for last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get predictions with actual results
            df = pd.read_sql_query('''
                SELECT * FROM prediction_storage 
                WHERE timestamp >= ? AND actual_position IS NOT NULL
            ''', conn, params=[cutoff_date])
        
        if df.empty:
            return {"error": "No completed predictions found in the specified period"}
        
        # Calculate metrics
        metrics = {
            "period_days": days,
            "total_predictions": len(df),
            "date_range": {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max()
            }
        }
        
        # MAE for each model type
        for model in ['rf_prediction', 'lstm_prediction', 'tabnet_prediction', 'feedforward_prediction', 'transformer_prediction', 'ensemble_alt_prediction', 'ensemble_prediction']:
            if model in df.columns:
                valid_predictions = df[df[model].notna()]
                if not valid_predictions.empty:
                    mae = np.mean(np.abs(valid_predictions[model] - valid_predictions['actual_position']))
                    metrics[f"{model}_mae"] = mae
        
        # Accuracy metrics (top 3 predictions)
        for model in ['rf_prediction', 'lstm_prediction', 'tabnet_prediction', 'feedforward_prediction', 'transformer_prediction', 'ensemble_alt_prediction', 'ensemble_prediction']:
            if model in df.columns:
                valid_predictions = df[df[model].notna()]
                if not valid_predictions.empty:
                    top3_accuracy = np.mean(valid_predictions[model] <= 3)
                    metrics[f"{model}_top3_accuracy"] = top3_accuracy
        
        return metrics
    
    def analyze_model_bias(self, days: int = 60) -> Dict[str, Any]:
        """Detect systematic biases by context"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM prediction_storage 
                WHERE timestamp >= ? AND actual_position IS NOT NULL
            ''', conn, params=[cutoff_date])
        
        if df.empty:
            return {"error": "No completed predictions found for bias analysis"}
        
        bias_analysis = {}
        
        # Bias by track condition
        if 'track_condition' in df.columns and df['track_condition'].notna().any():
            track_bias = df.groupby('track_condition').agg({
                'ensemble_prediction': 'mean',
                'actual_position': 'mean'
            }).round(3)
            bias_analysis['track_condition_bias'] = track_bias.to_dict()
        
        # Bias by weather
        if 'weather' in df.columns and df['weather'].notna().any():
            weather_bias = df.groupby('weather').agg({
                'ensemble_prediction': 'mean',
                'actual_position': 'mean'
            }).round(3)
            bias_analysis['weather_bias'] = weather_bias.to_dict()
        
        # Bias by field size
        if 'field_size' in df.columns and df['field_size'].notna().any():
            # Group field sizes into bins
            df['field_size_bin'] = pd.cut(df['field_size'], bins=[0, 8, 12, 16, 20], labels=['Small', 'Medium', 'Large', 'XLarge'])
            field_bias = df.groupby('field_size_bin').agg({
                'ensemble_prediction': 'mean',
                'actual_position': 'mean'
            }).round(3)
            bias_analysis['field_size_bias'] = field_bias.to_dict()
        
        # Bias by distance
        if 'distance' in df.columns and df['distance'].notna().any():
            # Group distances into bins
            df['distance_bin'] = pd.cut(df['distance'], bins=[0, 1200, 1600, 2000, 5000], labels=['Sprint', 'Mile', 'Classic', 'Distance'])
            distance_bias = df.groupby('distance_bin').agg({
                'ensemble_prediction': 'mean',
                'actual_position': 'mean'
            }).round(3)
            bias_analysis['distance_bias'] = distance_bias.to_dict()
        
        return bias_analysis
    
    def get_predictions_by_context(self, 
                                  race_type: Optional[str] = None,
                                  track_condition: Optional[str] = None,
                                  weather: Optional[str] = None,
                                  distance_range: Optional[Tuple[int, int]] = None,
                                  field_size_range: Optional[Tuple[int, int]] = None,
                                  days: int = 30) -> pd.DataFrame:
        """Filter predictions by race conditions"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = 'SELECT * FROM prediction_storage WHERE timestamp >= ?'
        params = [cutoff_date]
        
        if race_type:
            query += ' AND race_type = ?'
            params.append(race_type)
        
        if track_condition:
            query += ' AND track_condition = ?'
            params.append(track_condition)
        
        if weather:
            query += ' AND weather = ?'
            params.append(weather)
        
        if distance_range:
            query += ' AND distance BETWEEN ? AND ?'
            params.extend(distance_range)
        
        if field_size_range:
            query += ' AND field_size BETWEEN ? AND ?'
            params.extend(field_size_range)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df
    
    def get_training_feedback(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent predictions for incremental learning"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT race_id, horse_id, ensemble_prediction, actual_position,
                       model_versions, prediction_metadata
                FROM prediction_storage 
                WHERE timestamp >= ? AND actual_position IS NOT NULL
                ORDER BY timestamp DESC
            ''', [cutoff_date])
            
            feedback_data = []
            for row in cursor.fetchall():
                feedback_data.append({
                    'race_id': row[0],
                    'horse_id': row[1],
                    'predicted_position': row[2],
                    'actual_position': row[3],
                    'model_versions': json.loads(row[4]) if row[4] else {},
                    'metadata': json.loads(row[5]) if row[5] else {}
                })
        
        return feedback_data
    
    def export_for_analysis(self, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           output_path: Optional[str] = None) -> str:
        """Export data for external analysis"""
        query = 'SELECT * FROM prediction_storage WHERE 1=1'
        params = []
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp DESC'
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"analysis_output/prediction_export_{timestamp}.csv"
            Path(output_path).parent.mkdir(exist_ok=True)
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"Exported {len(df)} records to {output_path}")
        
        return output_path
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Archive or remove very old predictions"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            # Count records to be deleted
            cursor = conn.execute(
                'SELECT COUNT(*) FROM prediction_storage WHERE timestamp < ?',
                [cutoff_date]
            )
            count_to_delete = cursor.fetchone()[0]
            
            if count_to_delete > 0:
                # Export old data before deletion
                archive_path = self.export_for_analysis(
                    end_date=cutoff_date,
                    output_path=f"analysis_output/archived_predictions_{cutoff_date.strftime('%Y%m%d')}.csv"
                )
                
                # Delete old records
                conn.execute(
                    'DELETE FROM prediction_storage WHERE timestamp < ?',
                    [cutoff_date]
                )
                conn.commit()
                
                self.logger.info(f"Archived and deleted {count_to_delete} old records to {archive_path}")
            else:
                self.logger.info("No old records to cleanup")
        
        return count_to_delete
    
    def get_model_confidence_calibration(self, days: int = 30) -> Dict[str, Any]:
        """Analyze model confidence calibration"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT ensemble_prediction, ensemble_confidence_score, actual_position
                FROM prediction_storage 
                WHERE timestamp >= ? 
                AND actual_position IS NOT NULL 
                AND ensemble_confidence_score IS NOT NULL
            ''', conn, params=[cutoff_date])
        
        if df.empty:
            return {"error": "No predictions with confidence scores found"}
        
        # Bin confidence scores
        df['confidence_bin'] = pd.cut(df['ensemble_confidence_score'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        calibration_analysis = df.groupby('confidence_bin').agg({
            'ensemble_prediction': 'mean',
            'actual_position': 'mean',
            'ensemble_confidence_score': 'mean'
        }).round(3)
        
        # Calculate accuracy by confidence bin
        df['prediction_accurate'] = np.abs(df['ensemble_prediction'] - df['actual_position']) <= 1
        accuracy_by_confidence = df.groupby('confidence_bin')['prediction_accurate'].mean().round(3)
        
        return {
            'calibration_by_confidence': calibration_analysis.to_dict(),
            'accuracy_by_confidence': accuracy_by_confidence.to_dict(),
            'total_predictions_analyzed': len(df)
        }
    
    def suggest_blend_weights(self, days: int = 30) -> Dict[str, float]:
        """Analyze recent performance to suggest optimal blend weights"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT rf_prediction, lstm_prediction, tabnet_prediction, 
                       ensemble_prediction, actual_position
                FROM prediction_storage 
                WHERE timestamp >= ? AND actual_position IS NOT NULL
                AND rf_prediction IS NOT NULL 
                AND lstm_prediction IS NOT NULL 
                AND tabnet_prediction IS NOT NULL
            ''', conn, params=[cutoff_date])
        
        if df.empty:
            return {"error": "Insufficient data for blend weight optimization"}
        
        # Calculate MAE for each individual model
        rf_mae = np.mean(np.abs(df['rf_prediction'] - df['actual_position']))
        lstm_mae = np.mean(np.abs(df['lstm_prediction'] - df['actual_position']))
        tabnet_mae = np.mean(np.abs(df['tabnet_prediction'] - df['actual_position']))
        
        # Calculate inverse weights (better models get higher weights)
        total_inverse_mae = (1/rf_mae) + (1/lstm_mae) + (1/tabnet_mae)
        
        suggested_weights = {
            'rf_weight': round((1/rf_mae) / total_inverse_mae, 3),
            'lstm_weight': round((1/lstm_mae) / total_inverse_mae, 3),
            'tabnet_weight': round((1/tabnet_mae) / total_inverse_mae, 3)
        }
        
        # Add performance metrics for context
        suggested_weights['performance_metrics'] = {
            'rf_mae': round(rf_mae, 3),
            'lstm_mae': round(lstm_mae, 3),
            'tabnet_mae': round(tabnet_mae, 3),
            'current_ensemble_mae': round(np.mean(np.abs(df['ensemble_prediction'] - df['actual_position'])), 3)
        }
        
        return suggested_weights