#!/usr/bin/env python3
"""
Enhanced Competitive Analytics System

Leverages unused prediction tables to provide comprehensive competitive weighting
performance analysis and improved competitive field analysis.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging

class EnhancedCompetitiveAnalyzer:
    """
    Advanced competitive analysis system that leverages all prediction tables
    for enhanced competitive weighting performance analysis.
    """

    def __init__(self, db_path: str, verbose: bool = False):
        self.db_path = db_path
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)

        # Initialize tables for enhanced analysis
        self._init_enhanced_tables()

    def _init_enhanced_tables(self):
        """Ensure all prediction tables are optimized for analysis."""
        with sqlite3.connect(self.db_path) as conn:
            # Add performance indexes if they don't exist
            try:
                # Enhanced indexes for competitive analysis
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_competitive_strength
                    ON race_predictions (primary_advantage_type, advantage_strength)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_competitive_adjustment
                    ON race_predictions (competitive_adjustment)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prediction_performance
                    ON race_predictions (race_id, competitive_adjustment, advantage_strength)
                """)
                conn.commit()

                if self.verbose:
                    self.logger.info("Enhanced competitive analysis indexes created")

            except sqlite3.Error as e:
                self.logger.error(f"Error creating enhanced indexes: {e}")

    def store_enhanced_competitive_analysis(self, race_id: str, competitive_data: Dict) -> bool:
        """
        Store detailed competitive analysis in the competitive_analysis table.

        Args:
            race_id: Race identifier
            competitive_data: Detailed competitive analysis results

        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store in competitive_analysis table
                conn.execute("""
                    INSERT OR REPLACE INTO competitive_analysis (
                        id, race_prediction_id, field_statistics, top_contenders,
                        analysis_categories, adjustment_scaling, analysis_weights
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"{race_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    race_id,
                    json.dumps(competitive_data.get('field_statistics', {})),
                    json.dumps(competitive_data.get('top_contenders', [])),
                    json.dumps(competitive_data.get('analysis_categories', {})),
                    json.dumps(competitive_data.get('adjustment_scaling', {})),
                    json.dumps(competitive_data.get('analysis_weights', {}))
                ))
                conn.commit()

            if self.verbose:
                self.logger.info(f"Stored enhanced competitive analysis for race {race_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing enhanced competitive analysis: {e}")
            return False

    def analyze_competitive_performance(self, lookback_days: int = 30) -> Dict:
        """
        Comprehensive analysis of competitive weighting performance.

        Args:
            lookback_days: Days to look back for analysis

        Returns:
            Dict: Comprehensive performance analysis
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        with sqlite3.connect(self.db_path) as conn:
            # Get all recent predictions with competitive data
            df = pd.read_sql_query("""
                SELECT
                    race_id,
                    horse_id,
                    primary_advantage_type,
                    advantage_strength,
                    competitive_adjustment,
                    prediction_date
                FROM race_predictions
                WHERE prediction_date >= ?
                AND primary_advantage_type != 'none'
                AND advantage_strength != 0.0
                ORDER BY prediction_date DESC
            """, conn, params=(cutoff_date.isoformat(),))

        if df.empty:
            return {"error": "No competitive data found in the specified period"}

        analysis = {
            "period": f"Last {lookback_days} days",
            "total_predictions": len(df),
            "total_races": df['race_id'].nunique(),
            "competitive_predictions": len(df[df['primary_advantage_type'] != 'none']),
        }

        # Advantage type distribution analysis
        advantage_analysis = self._analyze_advantage_types(df)
        analysis['advantage_type_analysis'] = advantage_analysis

        # Strength distribution analysis
        strength_analysis = self._analyze_strength_distribution(df)
        analysis['strength_analysis'] = strength_analysis

        # Adjustment effectiveness analysis
        adjustment_analysis = self._analyze_adjustment_effectiveness(df)
        analysis['adjustment_analysis'] = adjustment_analysis

        # Field competitiveness analysis
        field_analysis = self._analyze_field_competitiveness(df)
        analysis['field_analysis'] = field_analysis

        return analysis

    def _analyze_advantage_types(self, df: pd.DataFrame) -> Dict:
        """Analyze the distribution and effectiveness of advantage types."""
        advantage_counts = df['primary_advantage_type'].value_counts()
        advantage_strengths = df.groupby('primary_advantage_type')['advantage_strength'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)

        return {
            "distribution": advantage_counts.to_dict(),
            "statistics": advantage_strengths.to_dict('index'),
            "most_common": advantage_counts.index[0] if len(advantage_counts) > 0 else None,
            "coverage_rate": len(df[df['primary_advantage_type'] != 'none']) / len(df) if len(df) > 0 else 0
        }

    def _analyze_strength_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze the distribution of advantage strengths."""
        strengths = df['advantage_strength']

        return {
            "mean": float(strengths.mean()),
            "std": float(strengths.std()),
            "median": float(strengths.median()),
            "min": float(strengths.min()),
            "max": float(strengths.max()),
            "quartiles": {
                "25%": float(strengths.quantile(0.25)),
                "75%": float(strengths.quantile(0.75))
            },
            "distribution": {
                "strong_positive": len(strengths[strengths > 0.5]),
                "moderate_positive": len(strengths[(strengths > 0) & (strengths <= 0.5)]),
                "neutral": len(strengths[strengths == 0]),
                "moderate_negative": len(strengths[(strengths < 0) & (strengths >= -0.5)]),
                "strong_negative": len(strengths[strengths < -0.5])
            }
        }

    def _analyze_adjustment_effectiveness(self, df: pd.DataFrame) -> Dict:
        """Analyze the effectiveness of competitive adjustments."""
        adjustments = df['competitive_adjustment']

        # Correlation between strength and adjustment
        correlation = df['advantage_strength'].corr(df['competitive_adjustment'])

        return {
            "mean_adjustment": float(adjustments.mean()),
            "std_adjustment": float(adjustments.std()),
            "adjustment_range": {
                "min": float(adjustments.min()),
                "max": float(adjustments.max())
            },
            "strength_correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            "adjustment_distribution": {
                "large_positive": len(adjustments[adjustments > 1.0]),
                "moderate_positive": len(adjustments[(adjustments > 0) & (adjustments <= 1.0)]),
                "neutral": len(adjustments[adjustments == 0]),
                "moderate_negative": len(adjustments[(adjustments < 0) & (adjustments >= -1.0)]),
                "large_negative": len(adjustments[adjustments < -1.0])
            }
        }

    def _analyze_field_competitiveness(self, df: pd.DataFrame) -> Dict:
        """Analyze competitiveness across different race fields."""
        race_stats = df.groupby('race_id').agg({
            'advantage_strength': ['count', 'mean', 'std'],
            'competitive_adjustment': ['mean', 'std']
        }).round(4)

        race_stats.columns = ['horses_count', 'avg_strength', 'strength_std',
                             'avg_adjustment', 'adjustment_std']

        return {
            "races_analyzed": len(race_stats),
            "avg_horses_per_race": float(race_stats['horses_count'].mean()),
            "field_competitiveness": {
                "avg_field_strength": float(race_stats['avg_strength'].mean()),
                "most_competitive_race": {
                    "race_id": race_stats['strength_std'].idxmax(),
                    "strength_variance": float(race_stats['strength_std'].max())
                },
                "least_competitive_race": {
                    "race_id": race_stats['strength_std'].idxmin(),
                    "strength_variance": float(race_stats['strength_std'].min())
                }
            }
        }

    def get_competitive_insights_for_race(self, race_id: str) -> Dict:
        """
        Get detailed competitive insights for a specific race.

        Args:
            race_id: Race identifier

        Returns:
            Dict: Detailed competitive insights
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get race predictions
            race_df = pd.read_sql_query("""
                SELECT
                    horse_id,
                    primary_advantage_type,
                    advantage_strength,
                    competitive_adjustment
                FROM race_predictions
                WHERE race_id = ?
                ORDER BY competitive_adjustment DESC
            """, conn, params=(race_id,))

            # Get any stored competitive analysis
            competitive_data = conn.execute("""
                SELECT field_statistics, top_contenders, analysis_categories
                FROM competitive_analysis
                WHERE race_prediction_id = ?
                ORDER BY id DESC
                LIMIT 1
            """, (race_id,)).fetchone()

        if race_df.empty:
            return {"error": f"No predictions found for race {race_id}"}

        insights = {
            "race_id": race_id,
            "total_horses": len(race_df),
            "competitive_horses": len(race_df[race_df['primary_advantage_type'] != 'none']),
        }

        # Advantage distribution in this race
        insights['advantage_distribution'] = race_df['primary_advantage_type'].value_counts().to_dict()

        # Top and bottom performers by competitive adjustment
        insights['top_contenders'] = race_df.head(3).to_dict('records')
        insights['bottom_contenders'] = race_df.tail(2).to_dict('records')

        # Field competitiveness metrics
        strengths = race_df['advantage_strength']
        insights['field_metrics'] = {
            "strength_range": float(strengths.max() - strengths.min()),
            "strength_variance": float(strengths.var()),
            "competitiveness_score": float(strengths.std()) if len(strengths) > 1 else 0.0
        }

        # Include stored competitive analysis if available
        if competitive_data:
            try:
                insights['stored_analysis'] = {
                    "field_statistics": json.loads(competitive_data[0]) if competitive_data[0] else {},
                    "top_contenders": json.loads(competitive_data[1]) if competitive_data[1] else [],
                    "analysis_categories": json.loads(competitive_data[2]) if competitive_data[2] else {}
                }
            except json.JSONDecodeError:
                pass

        return insights

    def migrate_predictions_to_enhanced_storage(self, limit: Optional[int] = None) -> Dict:
        """
        Migrate existing race_predictions to the enhanced prediction_results table.

        Args:
            limit: Optional limit on number of records to migrate

        Returns:
            Dict: Migration results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check current state - prediction_results is redundant, use race_predictions
                existing_count = conn.execute("SELECT COUNT(*) FROM race_predictions").fetchone()[0]
                race_pred_count = conn.execute("SELECT COUNT(*) FROM race_predictions").fetchone()[0]

                if existing_count >= race_pred_count:
                    return {
                        "status": "no_migration_needed",
                        "existing_enhanced": existing_count,
                        "source_records": race_pred_count
                    }

                # Prepare migration query
                limit_clause = f"LIMIT {limit}" if limit else ""

                # No migration needed - data is already in race_predictions table
                # This was redundant duplication - return existing count
                return {
                    "status": "no_migration_needed",
                    "message": "Data already exists in race_predictions table",
                    "existing_records": existing_count
                }

        except Exception as e:
            self.logger.error(f"Error during migration: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def generate_competitive_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive competitive analysis report.

        Args:
            output_file: Optional file path to save the report

        Returns:
            str: Report content
        """
        analysis = self.analyze_competitive_performance(lookback_days=30)

        if "error" in analysis:
            return f"Error generating report: {analysis['error']}"

        report = f"""
=== ENHANCED COMPETITIVE ANALYSIS REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: {analysis['period']}

OVERVIEW:
- Total predictions: {analysis['total_predictions']}
- Total races analyzed: {analysis['total_races']}
- Competitive predictions: {analysis['competitive_predictions']}
- Coverage rate: {analysis['advantage_type_analysis']['coverage_rate']:.2%}

ADVANTAGE TYPES:
"""

        for adv_type, count in analysis['advantage_type_analysis']['distribution'].items():
            stats = analysis['advantage_type_analysis']['statistics'][adv_type]
            report += f"- {adv_type}: {count} horses (avg strength: {stats['mean']:.3f})\n"

        report += f"""
STRENGTH DISTRIBUTION:
- Mean: {analysis['strength_analysis']['mean']:.3f}
- Range: {analysis['strength_analysis']['min']:.3f} to {analysis['strength_analysis']['max']:.3f}
- Strong advantages (>0.5): {analysis['strength_analysis']['distribution']['strong_positive']}
- Strong disadvantages (<-0.5): {analysis['strength_analysis']['distribution']['strong_negative']}

ADJUSTMENT EFFECTIVENESS:
- Mean adjustment: {analysis['adjustment_analysis']['mean_adjustment']:.3f}
- Strength correlation: {analysis['adjustment_analysis']['strength_correlation']:.3f}
- Adjustment range: {analysis['adjustment_analysis']['adjustment_range']['min']:.3f} to {analysis['adjustment_analysis']['adjustment_range']['max']:.3f}

FIELD ANALYSIS:
- Races analyzed: {analysis['field_analysis']['races_analyzed']}
- Avg horses per race: {analysis['field_analysis']['avg_horses_per_race']:.1f}
- Most competitive race: {analysis['field_analysis']['field_competitiveness']['most_competitive_race']['race_id']}
"""

        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report)
                report += f"\nReport saved to: {output_file}"
            except Exception as e:
                report += f"\nError saving report: {e}"

        return report