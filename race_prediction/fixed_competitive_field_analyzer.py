#!/usr/bin/env python3
"""
FIXED Competitive Field Analysis System

This is a corrected version that uses horse_id (idche) consistently throughout
instead of DataFrame index, fixing the critical horse identification bug.
"""

import numpy as np
import pandas as pd
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

class FixedCompetitiveFieldAnalyzer:
    """
    FIXED Competitive field analysis that correctly maps competitive data to horse IDs.

    Key fix: Uses horse_id (idche) consistently instead of DataFrame row index.
    """

    def __init__(self, db_path: str, verbose: bool = False):
        self.db_path = db_path
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Analysis weights for different competitive factors
        self.analysis_weights = {
            'speed_dominance': 0.3,      # Weight for speed analysis
            'track_specialization': 0.25, # Weight for track-specific performance
            'class_relief': 0.25,        # Weight for class level changes
            'form_momentum': 0.2         # Weight for recent form trends
        }

        # Thresholds for competitive advantages
        self.thresholds = {
            'speed_dominance_pct': 0.15,    # 15% faster than field average
            'track_win_rate_diff': 0.20,    # 20% higher win rate on track
            'class_drop_threshold': 0.10,    # 10% class drop advantage
            'form_trend_threshold': 0.25     # 25% form improvement trend
        }

    def analyze_competitive_field(self, base_predictions: Dict[str, np.ndarray],
                                 race_data: pd.DataFrame,
                                 race_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIXED: Analyze competitive dynamics using horse_id consistently.

        Args:
            base_predictions: Model predictions by name
            race_data: Race participant data with horse_id (idche)
            race_metadata: Race context information

        Returns:
            Competitive analysis results indexed by horse_id
        """
        if self.verbose:
            self.logger.info("Starting FIXED competitive field analysis using horse_id...")

        # Create horse_id mapping for consistent indexing
        horse_mapping = self._create_horse_id_mapping(race_data)

        if self.verbose:
            self.logger.info(f"Created horse mapping: {list(horse_mapping.keys())}")

        # Get historical data
        historical_data = self._get_historical_data_by_horse_ids(list(horse_mapping.keys()))

        # Step 1: Individual competitive analysis components (by horse_id)
        speed_analysis = self._analyze_speed_dominance_fixed(race_data, race_metadata, historical_data, horse_mapping)
        track_analysis = self._analyze_track_specialization_fixed(race_data, race_metadata, historical_data, horse_mapping)
        class_analysis = self._analyze_class_relief_fixed(race_data, race_metadata, historical_data, horse_mapping)
        form_analysis = self._analyze_form_momentum_fixed(race_data, race_metadata, historical_data, horse_mapping)

        # Step 2: Calculate composite competitive scores (by horse_id)
        competitive_scores = self._calculate_competitive_scores_fixed(
            speed_analysis, track_analysis, class_analysis, form_analysis, horse_mapping
        )

        # Step 3: Apply competitive adjustments to base predictions
        enhanced_predictions = {}
        adjustment_details = {}

        for model_name, predictions in base_predictions.items():
            enhanced_preds, adjustments = self._apply_competitive_adjustments_fixed(
                predictions, competitive_scores, race_data, horse_mapping
            )
            enhanced_predictions[model_name] = enhanced_preds
            adjustment_details[model_name] = adjustments

        return {
            'audit_trail': {
                'adjustment_summary': adjustment_details,
                'field_size': len(race_data),
                'analysis_timestamp': datetime.now().isoformat(),
                'horse_mapping': horse_mapping,  # Include for debugging
                'performance_expectations': self._calculate_performance_expectations_fixed(competitive_scores)
            },
            'competitive_analysis': {
                'competitive_scores': competitive_scores,  # Now indexed by horse_id
                'field_statistics': self._calculate_field_statistics_fixed(competitive_scores),
                'top_contenders': self._identify_top_contenders_fixed(competitive_scores, horse_mapping, top_n=3),
                'analysis_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'analysis_weights': self.analysis_weights,
                    'thresholds': self.thresholds,
                    'horses_analyzed': len(competitive_scores)
                }
            },
            'enhanced_predictions': enhanced_predictions
        }

    def _create_horse_id_mapping(self, race_data: pd.DataFrame) -> Dict[int, Dict]:
        """Create consistent mapping from horse_id to race data."""
        horse_mapping = {}

        for idx, row in race_data.iterrows():
            horse_id = row.get('idche')
            if pd.notna(horse_id):
                horse_id = int(horse_id)
                horse_mapping[horse_id] = {
                    'dataframe_index': idx,
                    'numero': row.get('numero', idx),
                    'cheval': row.get('cheval', f'Horse_{horse_id}'),
                    'row_data': row.to_dict()
                }

        return horse_mapping

    def _get_historical_data_by_horse_ids(self, horse_ids: List[int]) -> Dict[int, List[Dict]]:
        """Get historical data keyed by horse_id."""
        historical_data = {}

        if not horse_ids:
            return historical_data

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get historical data for these specific horses
                placeholders = ','.join('?' * len(horse_ids))
                cursor = conn.execute(f"""
                    SELECT participants FROM historical_races
                    WHERE participants IS NOT NULL
                    AND participants != '[]'
                    ORDER BY RANDOM() LIMIT 100
                """)

                for row in cursor.fetchall():
                    try:
                        participants = json.loads(row[0])
                        for participant in participants:
                            horse_id = participant.get('idche')
                            if horse_id and int(horse_id) in horse_ids:
                                horse_id = int(horse_id)
                                if horse_id not in historical_data:
                                    historical_data[horse_id] = []
                                historical_data[horse_id].append(participant)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue

        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error fetching historical data: {e}")

        return historical_data

    def _analyze_speed_dominance_fixed(self, race_data: pd.DataFrame, race_metadata: Dict[str, Any],
                                     historical_data: Dict[int, List[Dict]], horse_mapping: Dict[int, Dict]) -> Dict[int, Dict]:
        """FIXED: Speed analysis indexed by horse_id."""
        speed_analysis = {}

        # Extract best times by horse_id
        best_times_by_horse = {}

        for horse_id, horse_info in horse_mapping.items():
            if horse_id in historical_data:
                horse_history = historical_data[horse_id]
                best_time = None

                for race in horse_history:
                    record_g = race.get('recordG')
                    if record_g and record_g not in [None, '', 'null']:
                        try:
                            if isinstance(record_g, str) and "'" in record_g:
                                parts = record_g.replace('"', '').split("'")
                                if len(parts) == 2:
                                    minutes = int(parts[0])
                                    seconds = float(parts[1]) if parts[1] else 0
                                    total_seconds = minutes * 60 + seconds
                                    if best_time is None or total_seconds < best_time:
                                        best_time = total_seconds
                            elif isinstance(record_g, (int, float)) and record_g > 0:
                                if best_time is None or record_g < best_time:
                                    best_time = record_g
                        except (ValueError, TypeError):
                            continue

                best_times_by_horse[horse_id] = best_time

        # Calculate field statistics
        valid_times = [t for t in best_times_by_horse.values() if t is not None]

        if valid_times:
            field_avg_time = np.mean(valid_times)
            field_best_time = min(valid_times)
        else:
            field_avg_time = field_best_time = None

        # Analyze each horse
        for horse_id, horse_info in horse_mapping.items():
            best_time = best_times_by_horse.get(horse_id)

            if best_time and field_avg_time:
                speed_advantage_pct = (field_avg_time - best_time) / field_avg_time
                speed_score = max(-1.0, min(1.0, speed_advantage_pct * 3))  # Normalize to [-1, 1]
            else:
                speed_advantage_pct = 0.0
                speed_score = 0.0

            speed_analysis[horse_id] = {
                'best_time_sec': best_time,
                'field_best_time': field_best_time,
                'field_avg_time': field_avg_time,
                'speed_advantage_pct': speed_advantage_pct,
                'speed_score': speed_score,
                'has_speed_data': best_time is not None
            }

        return speed_analysis

    def _analyze_track_specialization_fixed(self, race_data: pd.DataFrame, race_metadata: Dict[str, Any],
                                          historical_data: Dict[int, List[Dict]], horse_mapping: Dict[int, Dict]) -> Dict[int, Dict]:
        """FIXED: Track specialization analysis indexed by horse_id."""
        track_analysis = {}
        current_track = race_metadata.get('hippo', 'Unknown')

        for horse_id, horse_info in horse_mapping.items():
            track_wins = track_starts = 0
            overall_wins = overall_starts = 0

            if horse_id in historical_data:
                for race in historical_data[horse_id]:
                    race_track = race.get('hippo', '')
                    position = race.get('derniereplace', '99')

                    # Try to parse position
                    try:
                        if isinstance(position, str) and position.isdigit():
                            pos = int(position)
                        elif isinstance(position, (int, float)):
                            pos = int(position)
                        else:
                            continue

                        overall_starts += 1
                        if pos == 1:
                            overall_wins += 1

                        if race_track == current_track:
                            track_starts += 1
                            if pos == 1:
                                track_wins += 1

                    except (ValueError, TypeError):
                        continue

            # Calculate track specialization metrics
            track_win_rate = track_wins / track_starts if track_starts > 0 else 0.0
            overall_win_rate = overall_wins / overall_starts if overall_starts > 0 else 0.0
            track_advantage = track_win_rate - overall_win_rate

            track_score = max(-1.0, min(1.0, track_advantage * 5))  # Normalize to [-1, 1]

            track_analysis[horse_id] = {
                'track_wins': track_wins,
                'track_starts': track_starts,
                'track_win_rate': track_win_rate,
                'overall_win_rate': overall_win_rate,
                'track_advantage': track_advantage,
                'track_score': track_score,
                'track_experience': track_starts >= 2
            }

        return track_analysis

    def _analyze_class_relief_fixed(self, race_data: pd.DataFrame, race_metadata: Dict[str, Any],
                                  historical_data: Dict[int, List[Dict]], horse_mapping: Dict[int, Dict]) -> Dict[int, Dict]:
        """FIXED: Class analysis indexed by horse_id."""
        class_analysis = {}

        for horse_id, horse_info in horse_mapping.items():
            row_data = horse_info['row_data']

            # Get class drop percentage if available
            class_drop = row_data.get('class_drop_pct', 0.0)
            if pd.isna(class_drop):
                class_drop = 0.0

            # Normalize class score
            class_score = max(-1.0, min(1.0, class_drop * 2))  # Normalize to [-1, 1]

            class_analysis[horse_id] = {
                'class_drop_pct': class_drop,
                'class_score': class_score,
                'has_class_relief': class_drop > self.thresholds['class_drop_threshold']
            }

        return class_analysis

    def _analyze_form_momentum_fixed(self, race_data: pd.DataFrame, race_metadata: Dict[str, Any],
                                   historical_data: Dict[int, List[Dict]], horse_mapping: Dict[int, Dict]) -> Dict[int, Dict]:
        """FIXED: Form analysis indexed by horse_id."""
        form_analysis = {}

        for horse_id, horse_info in horse_mapping.items():
            row_data = horse_info['row_data']

            # Get form indicators
            recent_positions = []
            if horse_id in historical_data:
                # Get last 5 races
                recent_races = historical_data[horse_id][-5:] if historical_data[horse_id] else []

                for race in recent_races:
                    pos = race.get('derniereplace', '99')
                    try:
                        if isinstance(pos, str) and pos.isdigit():
                            recent_positions.append(int(pos))
                        elif isinstance(pos, (int, float)):
                            recent_positions.append(int(pos))
                    except (ValueError, TypeError):
                        continue

            # Calculate form trend
            if len(recent_positions) >= 2:
                # Simple trend: improvement if recent average better than older average
                mid_point = len(recent_positions) // 2
                recent_avg = np.mean(recent_positions[-mid_point:])
                older_avg = np.mean(recent_positions[:-mid_point])
                form_trend = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0.0
            else:
                form_trend = 0.0

            form_score = max(-1.0, min(1.0, form_trend * 2))  # Normalize to [-1, 1]

            form_analysis[horse_id] = {
                'recent_positions': recent_positions,
                'form_trend': form_trend,
                'form_score': form_score,
                'races_analyzed': len(recent_positions)
            }

        return form_analysis

    def _calculate_competitive_scores_fixed(self, speed_analysis: Dict[int, Dict], track_analysis: Dict[int, Dict],
                                          class_analysis: Dict[int, Dict], form_analysis: Dict[int, Dict],
                                          horse_mapping: Dict[int, Dict]) -> Dict[int, Dict]:
        """FIXED: Calculate composite competitive scores indexed by horse_id."""
        competitive_scores = {}

        for horse_id in horse_mapping.keys():
            # Extract individual scores (default to 0 if not found)
            speed_score = speed_analysis.get(horse_id, {}).get('speed_score', 0.0)
            track_score = track_analysis.get(horse_id, {}).get('track_score', 0.0)
            class_score = class_analysis.get(horse_id, {}).get('class_score', 0.0)
            form_score = form_analysis.get(horse_id, {}).get('form_score', 0.0)

            # Calculate weighted composite score
            composite_score = (
                speed_score * self.analysis_weights['speed_dominance'] +
                track_score * self.analysis_weights['track_specialization'] +
                class_score * self.analysis_weights['class_relief'] +
                form_score * self.analysis_weights['form_momentum']
            )

            # Individual category advantages
            advantages = {
                'speed_advantage': speed_score > self.thresholds['speed_dominance_pct'],
                'track_advantage': track_score > self.thresholds['track_win_rate_diff'],
                'class_advantage': class_score > self.thresholds['class_drop_threshold'],
                'form_advantage': form_score > self.thresholds['form_trend_threshold']
            }

            total_advantages = sum(advantages.values())

            competitive_scores[horse_id] = {
                'speed_score': speed_score,
                'track_score': track_score,
                'class_score': class_score,
                'form_score': form_score,
                'composite_score': composite_score,
                'advantages': advantages,
                'total_advantages': total_advantages,
                'competitive_strength': self._categorize_competitive_strength(composite_score, total_advantages)
            }

        return competitive_scores

    def _apply_competitive_adjustments_fixed(self, predictions: np.ndarray, competitive_scores: Dict[int, Dict],
                                           race_data: pd.DataFrame, horse_mapping: Dict[int, Dict]) -> Tuple[np.ndarray, Dict]:
        """FIXED: Apply competitive adjustments using correct horse_id mapping."""
        adjusted_predictions = predictions.copy()
        adjustment_details = {}

        for i, horse_id in enumerate(horse_mapping.keys()):
            if horse_id in competitive_scores:
                horse_competitive = competitive_scores[horse_id]
                composite_score = horse_competitive['composite_score']

                # Scale adjustment based on competitive strength
                adjustment = composite_score * 2.0  # Adjust scale as needed
                adjusted_predictions[i] += adjustment

                adjustment_details[horse_id] = {
                    'total_adjustment': adjustment,
                    'components': {
                        'speed_contribution': horse_competitive['speed_score'] * self.analysis_weights['speed_dominance'] * 2.0,
                        'track_contribution': horse_competitive['track_score'] * self.analysis_weights['track_specialization'] * 2.0,
                        'class_contribution': horse_competitive['class_score'] * self.analysis_weights['class_relief'] * 2.0,
                        'form_contribution': horse_competitive['form_score'] * self.analysis_weights['form_momentum'] * 2.0
                    }
                }
            else:
                adjustment_details[horse_id] = {'total_adjustment': 0.0, 'components': {}}

        return adjusted_predictions, adjustment_details

    def _calculate_field_statistics_fixed(self, competitive_scores: Dict[int, Dict]) -> Dict[str, Any]:
        """Calculate field statistics from horse_id indexed scores."""
        if not competitive_scores:
            return {'field_competitiveness': 'unknown'}

        scores = [data['composite_score'] for data in competitive_scores.values()]

        return {
            'field_size': len(competitive_scores),
            'avg_competitive_score': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'score_range': float(max(scores) - min(scores)),
            'field_competitiveness': 'high' if np.std(scores) > 0.2 else 'moderate' if np.std(scores) > 0.1 else 'low'
        }

    def _identify_top_contenders_fixed(self, competitive_scores: Dict[int, Dict], horse_mapping: Dict[int, Dict], top_n: int = 3) -> List[Dict]:
        """Identify top contenders using horse_id."""
        sorted_horses = sorted(
            competitive_scores.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )

        contenders = []
        for horse_id, score_data in sorted_horses[:top_n]:
            horse_info = horse_mapping.get(horse_id, {})
            contenders.append({
                'horse_id': horse_id,
                'horse_name': horse_info.get('cheval', f'Horse_{horse_id}'),
                'numero': horse_info.get('numero', 0),
                'composite_score': score_data['composite_score'],
                'advantages': list(score_data['advantages'].keys()),
                'competitive_strength': score_data['competitive_strength']
            })

        return contenders

    def _calculate_performance_expectations_fixed(self, competitive_scores: Dict[int, Dict]) -> Dict[str, Any]:
        """Calculate performance expectations."""
        if not competitive_scores:
            return {}

        scores = [data['composite_score'] for data in competitive_scores.values()]

        return {
            'expected_leader': max(competitive_scores.items(), key=lambda x: x[1]['composite_score'])[0] if competitive_scores else None,
            'competitive_spread': float(max(scores) - min(scores)) if scores else 0.0,
            'field_balance': 'competitive' if np.std(scores) > 0.15 else 'dominated' if np.std(scores) < 0.05 else 'moderate'
        }

    def _categorize_competitive_strength(self, composite_score: float, total_advantages: int) -> str:
        """Categorize competitive strength."""
        if composite_score > 0.3 and total_advantages >= 3:
            return 'dominant'
        elif composite_score > 0.15 and total_advantages >= 2:
            return 'strong'
        elif composite_score > 0.0 and total_advantages >= 1:
            return 'moderate'
        elif composite_score > -0.15:
            return 'average'
        else:
            return 'weak'