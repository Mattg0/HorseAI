"""
Competitive Field Analyzer for Horse Race Predictions

This module implements competitive weighting analysis to enhance base model predictions
by comparing each horse's advantages against the field in key performance categories:

1. Speed Dominance: Historical best times vs field
2. Track Specialization: Venue-specific performance advantages
3. Class Relief: Favorable class movements vs competition
4. Form Momentum: Superior recent trend trajectories vs field

Target: Improve R² from 0.2547 to 0.27+ through intelligent competitive adjustments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
import sqlite3
from pathlib import Path


class CompetitiveFieldAnalyzer:
    """
    Analyzes competitive advantages within a race field and applies intelligent
    weighting to base model predictions.
    """

    def __init__(self, verbose: bool = False, db_path: str = None):
        """
        Initialize the competitive field analyzer.

        Args:
            verbose: Enable detailed logging
            db_path: Path to database for historical data queries
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Database path for historical data queries
        if db_path is None:
            from utils.env_setup import AppConfig
            config = AppConfig()
            self.db_path = config.get_sqlite_dbpath(config._config.base.active_db)
        else:
            self.db_path = db_path

        # Competitive analysis weights (tunable parameters)
        self.analysis_weights = {
            'speed_dominance': 0.30,      # Speed advantage weight
            'track_specialization': 0.25,  # Venue advantage weight
            'class_relief': 0.25,         # Class movement advantage
            'form_momentum': 0.20          # Recent form trend advantage
        }

        # Adjustment scaling factors
        self.adjustment_scaling = {
            'speed_dominance': 0.15,       # Max ±15% adjustment for speed
            'track_specialization': 0.12,  # Max ±12% adjustment for track
            'class_relief': 0.10,         # Max ±10% adjustment for class
            'form_momentum': 0.08          # Max ±8% adjustment for form
        }

        # Performance thresholds for significant advantages
        self.thresholds = {
            'speed_dominance_pct': 0.05,   # 5% speed advantage threshold
            'track_win_rate_diff': 0.15,   # 15% win rate difference
            'class_drop_threshold': 0.20,  # 20% class drop advantage
            'form_trend_threshold': 0.10   # 10% form trend advantage
        }

    def _get_historical_horse_data(self, horse_ids: List[int], limit_per_horse: int = 20) -> Dict[int, List[Dict]]:
        """
        Retrieve historical performance data for horses from the historical_races table using batch queries.

        Args:
            horse_ids: List of horse IDs (idche values) to query
            limit_per_horse: Maximum number of historical races per horse

        Returns:
            Dictionary mapping horse_id to list of historical race data
        """
        historical_data = {horse_id: [] for horse_id in horse_ids}

        if not horse_ids:
            return historical_data

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # Set connection timeout and performance pragmas
            conn.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout
            conn.execute("PRAGMA cache_size = 10000")   # Larger cache

            if self.verbose:
                self.logger.info(f"Batch querying historical data for {len(horse_ids)} horses: {horse_ids}")

            # Create placeholder string for IN clause
            placeholders = ','.join('?' * len(horse_ids))

            # Single batch query to get all historical data for all horses
            batch_query = f"""
            WITH horse_races AS (
                SELECT hr.comp, hr.jour, hr.hippo, hr.dist, hr.typec,
                       json_extract(participant.value, '$.idche') as horse_id,
                       participant.value as participant_data,
                       ROW_NUMBER() OVER (
                           PARTITION BY json_extract(participant.value, '$.idche')
                           ORDER BY hr.jour DESC
                       ) as rn
                FROM historical_races hr,
                     json_each(hr.participants) as participant
                WHERE json_extract(participant.value, '$.idche') IN ({placeholders})
                AND hr.jour >= date('now', '-365 days')
                AND json_extract(participant.value, '$.idche') IS NOT NULL
            )
            SELECT comp, jour, hippo, dist, typec, horse_id, participant_data
            FROM horse_races
            WHERE rn <= ?
            ORDER BY horse_id, jour DESC
            """

            if self.verbose:
                import time
                query_start = time.time()

            # Execute batch query with all horse IDs plus limit
            cursor = conn.execute(batch_query, horse_ids + [limit_per_horse])
            rows = cursor.fetchall()

            if self.verbose:
                query_time = time.time() - query_start
                self.logger.info(f"Batch query completed in {query_time:.2f}s, found {len(rows)} total records")

            # Process results and group by horse_id
            for row in rows:
                try:
                    horse_id = int(row['horse_id'])
                    if horse_id not in historical_data:
                        continue

                    # Parse the participant data
                    horse_data = json.loads(row['participant_data'])

                    if horse_data:
                        race_data = {
                            'race_id': row['comp'],
                            'date': row['jour'],
                            'venue': row['hippo'],
                            'distance': row['dist'],
                            'race_type': row['typec'],
                            # Horse-specific data from JSON
                            'recordG': horse_data.get('recordG'),
                            'victories': horse_data.get('victoirescheval'),
                            'places': horse_data.get('placescheval'),
                            'total_races': horse_data.get('coursescheval'),
                            'earnings': horse_data.get('gainsCarriere'),
                            'last_position': horse_data.get('derniereplace'),
                            'race_venue': horse_data.get('hippo', row['hippo'])
                        }
                        historical_data[horse_id].append(race_data)

                except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
                    if self.verbose:
                        self.logger.warning(f"Error parsing participants JSON: {e}")
                    continue

            conn.close()

            if self.verbose:
                total_historical_races = sum(len(races) for races in historical_data.values())
                self.logger.info(f"Batch query retrieved {total_historical_races} total historical races")
                self.logger.info(f"=== COMPETITIVE RACE COUNTS PER HORSE ===")
                for horse_id, races in historical_data.items():
                    if races:
                        self.logger.info(f"🏇 Horse {horse_id}: {len(races)} historical races found")
                    else:
                        self.logger.info(f"❌ Horse {horse_id}: No historical data found")

        except Exception as e:
            self.logger.error(f"Error retrieving historical horse data: {str(e)}")
            # Return empty dict on error
            historical_data = {horse_id: [] for horse_id in horse_ids}

        return historical_data

    def analyze_competitive_field(self, race_data: pd.DataFrame,
                                base_predictions: Dict[str, np.ndarray],
                                race_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze competitive advantages and apply weighted adjustments to base predictions.

        Args:
            race_data: DataFrame containing all horses in the race with features
            base_predictions: Dictionary of base model predictions {model_name: predictions}
            race_metadata: Race context (distance, venue, class, etc.)

        Returns:
            Dictionary containing:
            - enhanced_predictions: Adjusted predictions with competitive weighting
            - competitive_analysis: Detailed analysis for each horse
            - adjustment_summary: Summary of adjustments applied
            - audit_trail: Complete analysis audit trail
        """
        if self.verbose:
            self.logger.info(f"=== COMPETITIVE ANALYSIS DEBUG ===")
            self.logger.info(f"Analyzing competitive field for race with {len(race_data)} horses")
            self.logger.info(f"Race data shape: {race_data.shape}")
            self.logger.info(f"Available columns: {list(race_data.columns)}")

            # Debug: Check for key competitive analysis columns
            critical_cols = ['recordG', 'hippo', 'coursescheval', 'victoirescheval', 'placescheval', 'gainsCarriere', 'age', 'derniereplace']
            self.logger.info(f"=== CRITICAL COLUMNS ANALYSIS ===")
            for col in critical_cols:
                if col in race_data.columns:
                    non_null_count = race_data[col].notna().sum()
                    total_count = len(race_data)
                    self.logger.info(f"✅ {col}: {non_null_count}/{total_count} non-null values ({non_null_count/total_count*100:.1f}%)")
                    if non_null_count > 0:
                        sample_values = race_data[col].dropna().head(3).tolist()
                        self.logger.info(f"   Sample values: {sample_values}")
                else:
                    self.logger.info(f"❌ {col}: COLUMN NOT FOUND")

            # Show race metadata for context
            self.logger.info(f"=== RACE METADATA ===")
            for key, value in race_metadata.items():
                self.logger.info(f"{key}: {value}")

            # Show sample of race data
            self.logger.info(f"=== SAMPLE RACE DATA ===")
            if len(race_data) > 0:
                sample_cols = ['numero', 'cheval', 'recordG', 'hippo', 'coursescheval'] if any(col in race_data.columns for col in ['numero', 'cheval', 'recordG', 'hippo', 'coursescheval']) else list(race_data.columns)[:5]
                available_sample_cols = [col for col in sample_cols if col in race_data.columns]
                if available_sample_cols:
                    self.logger.info(f"Sample data (columns: {available_sample_cols}):")
                    for i, row in race_data[available_sample_cols].head(3).iterrows():
                        self.logger.info(f"  Horse {i}: {dict(row)}")

            self.logger.info(f"=== END COMPETITIVE ANALYSIS DEBUG ===\n")

        # Step 0: Retrieve historical data for all horses
        horse_ids = []
        if 'idche' in race_data.columns:
            horse_ids = [int(h) for h in race_data['idche'] if pd.notna(h) and h != 0]

        historical_data = {}
        if horse_ids:
            if self.verbose:
                self.logger.info(f"=== RETRIEVING HISTORICAL DATA ===")
                self.logger.info(f"Fetching historical data for {len(horse_ids)} horses: {horse_ids}")

            # Use optimized historical data retrieval with JSON extraction
            historical_data = self._get_historical_horse_data(horse_ids)

            if self.verbose:
                total_historical_races = sum(len(races) for races in historical_data.values())
                self.logger.info(f"Retrieved {total_historical_races} total historical races")
                self.logger.info(f"=== COMPETITIVE RACE COUNTS BY HORSE ===")
                for horse_id, races in historical_data.items():
                    if races:
                        self.logger.info(f"🏇 Horse {horse_id}: {len(races)} historical races")
                    else:
                        self.logger.info(f"❌ Horse {horse_id}: No historical data found")
                self.logger.info(f"=== END HISTORICAL DATA RETRIEVAL ===\n")
        else:
            if self.verbose:
                self.logger.info("⚠️ No valid horse IDs found for historical data retrieval")

        # Step 1: Analyze each competitive category using reliable data only
        earnings_analysis = self._analyze_earnings_quality(race_data, race_metadata)
        track_analysis = self._analyze_track_specialization(race_data, race_metadata, historical_data)
        class_analysis = self._analyze_class_relief(race_data, race_metadata, historical_data)
        form_analysis = self._analyze_form_momentum(race_data, race_metadata, historical_data)
        connection_analysis = self._analyze_hot_connections(race_data, race_metadata)
        distance_analysis = self._analyze_distance_comfort(race_data, race_metadata)

        # Step 2: Calculate realistic composite competitive scores
        competitive_scores = self._calculate_realistic_competitive_scores(
            earnings_analysis, track_analysis, form_analysis, connection_analysis, distance_analysis
        )

        # Step 3: Apply competitive adjustments to base predictions
        enhanced_predictions = {}
        adjustment_details = {}

        for model_name, predictions in base_predictions.items():
            enhanced_preds, adjustments = self._apply_realistic_competitive_adjustments(
                predictions, competitive_scores, race_data
            )
            enhanced_predictions[model_name] = enhanced_preds
            adjustment_details[model_name] = adjustments

        # Step 4: Create comprehensive analysis output
        competitive_analysis = self._create_realistic_competitive_analysis(
            race_data, earnings_analysis, track_analysis, form_analysis,
            connection_analysis, distance_analysis, competitive_scores
        )

        # Step 5: Generate audit trail
        audit_trail = self._generate_audit_trail(
            race_data, base_predictions, enhanced_predictions,
            competitive_analysis, adjustment_details, race_metadata
        )

        # Add historical race counts to the final output
        historical_race_counts = {}
        if historical_data:
            for horse_id, races in historical_data.items():
                historical_race_counts[horse_id] = len(races)

            if self.verbose:
                total_competitive_races = sum(historical_race_counts.values())
                horses_with_data = sum(1 for count in historical_race_counts.values() if count > 0)
                avg_races_per_horse = total_competitive_races / len(historical_race_counts) if historical_race_counts else 0
                self.logger.info(f"📊 COMPETITIVE ANALYSIS SUMMARY:")
                self.logger.info(f"   Total competitive races: {total_competitive_races}")
                self.logger.info(f"   Horses with historical data: {horses_with_data}/{len(historical_race_counts)}")
                self.logger.info(f"   Average races per horse: {avg_races_per_horse:.1f}")

        return {
            'enhanced_predictions': enhanced_predictions,
            'competitive_analysis': competitive_analysis,
            'adjustment_summary': self._create_adjustment_summary(adjustment_details),
            'audit_trail': audit_trail,
            'historical_race_counts': historical_race_counts
        }

    def _analyze_earnings_quality(self, race_data: pd.DataFrame,
                                 race_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze competitive advantages using earnings per race as quality proxy.
        Uses only reliable data: gainsCarriere and coursescheval.

        Args:
            race_data: Race participant data
            race_metadata: Race context including purse

        Returns:
            Earnings quality analysis for each horse
        """
        earnings_analysis = {}

        # Calculate earnings per race for each horse
        earnings_per_race = {}
        valid_horses = []

        for idx in race_data.index:
            gains = race_data.loc[idx, 'gainsCarriere']
            courses = race_data.loc[idx, 'coursescheval']

            # Only use horses with reliable earnings data
            if pd.notna(gains) and pd.notna(courses) and gains > 0 and courses > 0:
                epr = gains / courses
                earnings_per_race[idx] = epr
                valid_horses.append(idx)
            else:
                earnings_per_race[idx] = 0.0

        if self.verbose:
            self.logger.info(f"=== EARNINGS QUALITY ANALYSIS ===")
            self.logger.info(f"Valid earnings data: {len(valid_horses)}/{len(race_data)} horses")
            if valid_horses:
                sample_epr = [earnings_per_race[idx] for idx in valid_horses[:3]]
                self.logger.info(f"Sample earnings per race: {[f'{epr:,.0f}' for epr in sample_epr]}")

        # Calculate field statistics
        if valid_horses:
            valid_earnings = [earnings_per_race[idx] for idx in valid_horses]
            field_avg_earnings = np.mean(valid_earnings)
            field_max_earnings = max(valid_earnings)
            field_std_earnings = np.std(valid_earnings) if len(valid_earnings) > 1 else 0.0

            if self.verbose:
                self.logger.info(f"Field earnings stats: avg={field_avg_earnings:,.0f}, max={field_max_earnings:,.0f}")
        else:
            field_avg_earnings = 0.0
            field_max_earnings = 0.0
            field_std_earnings = 0.0

        # Analyze each horse
        for idx in race_data.index:
            horse_earnings = earnings_per_race[idx]

            if horse_earnings == 0.0 or field_avg_earnings == 0.0:
                # No reliable data - neutral score
                earnings_analysis[idx] = {
                    'earnings_per_race': horse_earnings,
                    'field_avg_earnings': field_avg_earnings,
                    'earnings_advantage_pct': 0.0,
                    'earnings_rank': len(race_data) // 2,
                    'earnings_score': 0.0,
                    'class_advantage': False
                }
            else:
                # Calculate earnings advantage
                earnings_advantage_pct = (horse_earnings - field_avg_earnings) / field_avg_earnings

                # Calculate rank (higher earnings = better)
                earnings_rank = sum(1 for e in earnings_per_race.values() if e >= horse_earnings)

                # Normalized earnings score (-1 to +1)
                if field_std_earnings > 0:
                    earnings_score = (horse_earnings - field_avg_earnings) / field_std_earnings
                    earnings_score = np.clip(earnings_score, -2.0, 2.0) / 2.0
                else:
                    earnings_score = 0.0

                # Detect class advantage (high earner in lower purse race)
                race_purse = race_metadata.get('cheque', race_metadata.get('purse', 100000))
                purse_per_starter = race_purse / max(len(race_data), 1) if race_purse > 0 else 0

                class_advantage = False
                if purse_per_starter > 0 and horse_earnings > purse_per_starter * 2:
                    class_advantage = True  # Horse typically earns 2x+ today's expected payout

                earnings_analysis[idx] = {
                    'earnings_per_race': horse_earnings,
                    'field_avg_earnings': field_avg_earnings,
                    'earnings_advantage_pct': earnings_advantage_pct,
                    'earnings_rank': earnings_rank,
                    'earnings_score': earnings_score,
                    'class_advantage': class_advantage
                }

        return earnings_analysis

    def _analyze_track_specialization(self, race_data: pd.DataFrame,
                                    race_metadata: Dict[str, Any],
                                    historical_data: Dict[int, List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze track specialization using venue-specific performance data.
        Uses reliable data: pourcVictChevalHippo, nbrCourseChevalHippo.

        Args:
            race_data: Race participant data
            race_metadata: Race context including venue

        Returns:
            Track specialization analysis for each horse
        """
        track_analysis = {}

        # Get venue-specific data with sample size validation
        venue_win_pcts = {}
        venue_sample_sizes = {}
        reliable_horses = []

        for idx in race_data.index:
            win_pct = race_data.loc[idx, 'pourcVictChevalHippo']
            sample_size = race_data.loc[idx, 'nbrCourseChevalHippo'] if 'nbrCourseChevalHippo' in race_data.columns else 0

            # Only trust horses with minimum sample size at venue
            if pd.notna(win_pct) and pd.notna(sample_size) and sample_size >= 3:
                venue_win_pcts[idx] = win_pct
                venue_sample_sizes[idx] = sample_size
                reliable_horses.append(idx)
            else:
                venue_win_pcts[idx] = 0.0
                venue_sample_sizes[idx] = 0

        if self.verbose:
            self.logger.info(f"=== TRACK SPECIALIZATION ANALYSIS ===")
            self.logger.info(f"Horses with reliable venue data (3+ races): {len(reliable_horses)}/{len(race_data)}")
            if reliable_horses:
                sample_data = [(idx, venue_win_pcts[idx], venue_sample_sizes[idx]) for idx in reliable_horses[:3]]
                for idx, win_pct, sample in sample_data:
                    self.logger.info(f"  Horse {idx}: {win_pct:.1f}% wins in {sample} venue races")

        # Calculate field statistics from reliable horses only
        if reliable_horses:
            reliable_win_pcts = [venue_win_pcts[idx] for idx in reliable_horses]
            field_avg_win_pct = np.mean(reliable_win_pcts)
            field_max_win_pct = max(reliable_win_pcts)
            field_std_win_pct = np.std(reliable_win_pcts) if len(reliable_win_pcts) > 1 else 0.0

            if self.verbose:
                self.logger.info(f"Field venue stats: avg={field_avg_win_pct:.1f}%, max={field_max_win_pct:.1f}%")
        else:
            field_avg_win_pct = 0.0
            field_max_win_pct = 0.0
            field_std_win_pct = 0.0

        # Analyze each horse
        for idx in race_data.index:
            horse_win_pct = venue_win_pcts[idx]
            horse_sample_size = venue_sample_sizes[idx]

            if horse_sample_size < 3 or field_avg_win_pct == 0.0:
                # Insufficient data - neutral score
                track_analysis[idx] = {
                    'venue_win_pct': horse_win_pct,
                    'venue_sample_size': horse_sample_size,
                    'field_avg_win_pct': field_avg_win_pct,
                    'track_advantage_pct': 0.0,
                    'track_specialist': False,
                    'track_score': 0.0,
                    'reliable_data': False
                }
            else:
                # Calculate track advantage
                track_advantage_pct = (horse_win_pct - field_avg_win_pct) / max(field_avg_win_pct, 0.01)

                # Detect track specialist (1.5x above field average with good sample)
                track_specialist = (horse_win_pct > field_avg_win_pct * 1.5 and
                                  horse_sample_size >= 5 and
                                  horse_win_pct > 10.0)  # At least 10% win rate

                # Normalized track score (-1 to +1)
                if field_std_win_pct > 0:
                    track_score = (horse_win_pct - field_avg_win_pct) / field_std_win_pct
                    track_score = np.clip(track_score, -2.0, 2.0) / 2.0
                else:
                    track_score = 0.0

                track_analysis[idx] = {
                    'venue_win_pct': horse_win_pct,
                    'venue_sample_size': horse_sample_size,
                    'field_avg_win_pct': field_avg_win_pct,
                    'track_advantage_pct': track_advantage_pct,
                    'track_specialist': track_specialist,
                    'track_score': track_score,
                    'reliable_data': True
                }

        return track_analysis

    def _analyze_class_relief(self, race_data: pd.DataFrame,
                            race_metadata: Dict[str, Any],
                            historical_data: Dict[int, List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze class movement advantages using class_drop_pct and purse_ratio.

        Args:
            race_data: Race participant data
            race_metadata: Race context

        Returns:
            Class relief analysis for each horse
        """
        class_analysis = {}

        # Extract class movement indicators
        class_drop_pct = race_data.get('class_drop_pct', pd.Series([0.0] * len(race_data)))
        purse_ratio = race_data.get('purse_ratio', pd.Series([1.0] * len(race_data)))

        # Calculate field statistics
        valid_drops = class_drop_pct.fillna(0.0)
        valid_ratios = purse_ratio.fillna(1.0)

        field_avg_drop = valid_drops.mean()
        field_avg_ratio = valid_ratios.mean()
        field_std_drop = valid_drops.std()

        for idx in race_data.index:
            horse_drop_pct = class_drop_pct.get(idx, 0.0)
            horse_purse_ratio = purse_ratio.get(idx, 1.0)

            if pd.isna(horse_drop_pct):
                horse_drop_pct = 0.0
            if pd.isna(horse_purse_ratio):
                horse_purse_ratio = 1.0

            # Calculate class relief advantage (positive drop = easier class)
            class_advantage = horse_drop_pct - field_avg_drop

            # Factor in purse ratio (lower purse = easier race)
            purse_advantage = (field_avg_ratio - horse_purse_ratio) / max(field_avg_ratio, 0.01)

            # Combined class relief score
            class_relief_score = (class_advantage * 0.7) + (purse_advantage * 0.3)

            # Calculate rank (higher drop = better)
            class_rank = (valid_drops >= horse_drop_pct).sum()

            # Normalize class score (-1 to +1)
            if field_std_drop > 0:
                normalized_score = class_advantage / field_std_drop
                normalized_score = np.clip(normalized_score, -2.0, 2.0) / 2.0
            else:
                normalized_score = 0.0

            class_analysis[idx] = {
                'class_drop_pct': horse_drop_pct,
                'purse_ratio': horse_purse_ratio,
                'class_advantage': class_advantage,
                'class_relief_score': class_relief_score,
                'class_rank': class_rank,
                'class_score': normalized_score
            }

        return class_analysis

    def _analyze_form_momentum(self, race_data: pd.DataFrame,
                             race_metadata: Dict[str, Any],
                             historical_data: Dict[int, List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze form momentum using reliable trend and performance data.
        Uses: che_global_trend, che_global_recent_perf, joc_global_trend.

        Args:
            race_data: Race participant data
            race_metadata: Race context

        Returns:
            Form momentum analysis for each horse
        """
        form_analysis = {}

        # Extract reliable form indicators
        che_trend = race_data.get('che_global_trend', pd.Series([0.0] * len(race_data))).fillna(0.0)
        che_recent_perf = race_data.get('che_global_recent_perf', pd.Series([0.0] * len(race_data))).fillna(0.0)
        joc_trend = race_data.get('joc_global_trend', pd.Series([0.0] * len(race_data))).fillna(0.0)

        # Calculate field statistics
        field_avg_che_trend = che_trend.mean()
        field_avg_che_perf = che_recent_perf.mean()
        field_avg_joc_trend = joc_trend.mean()

        field_std_che_trend = che_trend.std()
        field_std_che_perf = che_recent_perf.std()
        field_std_joc_trend = joc_trend.std()

        if self.verbose:
            self.logger.info(f"=== FORM MOMENTUM ANALYSIS ===")
            self.logger.info(f"Field avg horse trend: {field_avg_che_trend:.3f}, recent perf: {field_avg_che_perf:.2f}")
            self.logger.info(f"Field avg jockey trend: {field_avg_joc_trend:.3f}")

        # Identify horses with positive momentum
        improving_horses = []
        for idx in race_data.index:
            horse_che_trend = che_trend.loc[idx]
            horse_che_perf = che_recent_perf.loc[idx]

            # Improving form: positive trend AND above-average recent performance
            if horse_che_trend > 0 and horse_che_perf > field_avg_che_perf:
                improving_horses.append(idx)

        if self.verbose and improving_horses:
            self.logger.info(f"Horses with improving form: {len(improving_horses)}/{len(race_data)}")

        # Analyze each horse
        for idx in race_data.index:
            horse_che_trend = che_trend.loc[idx]
            horse_che_perf = che_recent_perf.loc[idx]
            horse_joc_trend = joc_trend.loc[idx]

            # Calculate advantages
            che_trend_advantage = horse_che_trend - field_avg_che_trend
            che_perf_advantage = horse_che_perf - field_avg_che_perf
            joc_trend_advantage = horse_joc_trend - field_avg_joc_trend

            # Detect positive momentum (improving horse with good recent form)
            positive_momentum = (horse_che_trend > 0.01 and  # Positive trend
                               horse_che_perf > field_avg_che_perf and  # Above avg recent perf
                               idx in improving_horses)

            # Combined form score
            che_trend_norm = (che_trend_advantage / field_std_che_trend) if field_std_che_trend > 0 else 0.0
            che_perf_norm = (che_perf_advantage / field_std_che_perf) if field_std_che_perf > 0 else 0.0
            joc_trend_norm = (joc_trend_advantage / field_std_joc_trend) if field_std_joc_trend > 0 else 0.0

            # Weight: recent performance (40%), horse trend (35%), jockey trend (25%)
            form_score = (che_perf_norm * 0.4) + (che_trend_norm * 0.35) + (joc_trend_norm * 0.25)
            form_score = np.clip(form_score, -2.0, 2.0) / 2.0  # Normalize to [-1, 1]

            # Overall form momentum
            form_momentum = (che_trend_advantage * 0.5) + (che_perf_advantage * 0.3) + (joc_trend_advantage * 0.2)

            form_analysis[idx] = {
                'che_global_trend': horse_che_trend,
                'che_global_recent_perf': horse_che_perf,
                'joc_global_trend': horse_joc_trend,
                'che_trend_advantage': che_trend_advantage,
                'che_perf_advantage': che_perf_advantage,
                'joc_trend_advantage': joc_trend_advantage,
                'positive_momentum': positive_momentum,
                'form_momentum': form_momentum,
                'form_score': form_score
            }

        return form_analysis

    def _analyze_hot_connections(self, race_data: pd.DataFrame,
                               race_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze same-day momentum for jockey/trainer connections.
        Uses: victoirejockeyjour, victoireentraineurjour, montesdujockeyjour.

        Args:
            race_data: Race participant data
            race_metadata: Race context

        Returns:
            Hot connection analysis for each horse
        """
        connection_analysis = {}

        # Extract same-day performance data
        jockey_wins = race_data.get('victoirejockeyjour', pd.Series([0] * len(race_data))).fillna(0)
        trainer_wins = race_data.get('victoireentraineurjour', pd.Series([0] * len(race_data))).fillna(0)
        jockey_rides = race_data.get('montesdujockeyjour', pd.Series([1] * len(race_data))).fillna(1)

        if self.verbose:
            self.logger.info(f"=== HOT CONNECTIONS ANALYSIS ===")
            hot_jockeys = sum(1 for w in jockey_wins if w > 0)
            hot_trainers = sum(1 for w in trainer_wins if w > 0)
            self.logger.info(f"Jockeys with wins today: {hot_jockeys}/{len(race_data)}")
            self.logger.info(f"Trainers with wins today: {hot_trainers}/{len(race_data)}")

        # Analyze each horse
        for idx in race_data.index:
            j_wins = int(jockey_wins.loc[idx])
            t_wins = int(trainer_wins.loc[idx])
            j_rides = max(int(jockey_rides.loc[idx]), 1)

            # Hot jockey detection (wins today, not overworked)
            hot_jockey = j_wins > 0 and j_rides <= 4  # Wins but not too many rides

            # Hot trainer detection
            hot_trainer = t_wins > 0

            # Hot connection (both hot)
            hot_connection = hot_jockey and hot_trainer

            # Fatigue penalty for overworked jockey
            jockey_fatigue = j_rides >= 5

            # Connection momentum score
            jockey_score = j_wins - (0.1 * max(j_rides - 4, 0))  # Penalty for 5+ rides
            trainer_score = t_wins
            connection_score = (jockey_score * 0.6) + (trainer_score * 0.4)

            connection_analysis[idx] = {
                'jockey_wins_today': j_wins,
                'trainer_wins_today': t_wins,
                'jockey_rides_today': j_rides,
                'hot_jockey': hot_jockey,
                'hot_trainer': hot_trainer,
                'hot_connection': hot_connection,
                'jockey_fatigue': jockey_fatigue,
                'connection_score': connection_score
            }

        return connection_analysis

    def _analyze_distance_comfort(self, race_data: pd.DataFrame,
                                race_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze distance patterns and comfort.
        Uses: dist (current), dernieredist (last), distance experience.

        Args:
            race_data: Race participant data
            race_metadata: Race context including current distance

        Returns:
            Distance comfort analysis for each horse
        """
        distance_analysis = {}

        current_distance = race_metadata.get('dist', race_metadata.get('distance', 2000))

        # Get last race distances
        last_distances = race_data.get('dernieredist', pd.Series([0] * len(race_data))).fillna(0)

        if self.verbose:
            self.logger.info(f"=== DISTANCE COMFORT ANALYSIS ===")
            self.logger.info(f"Current race distance: {current_distance}m")
            distance_matches = sum(1 for d in last_distances if abs(d - current_distance) <= 200)
            self.logger.info(f"Horses with similar recent distance: {distance_matches}/{len(race_data)}")

        # Analyze each horse
        for idx in race_data.index:
            last_dist = last_distances.loc[idx]

            # Distance change analysis
            distance_change = abs(current_distance - last_dist) if last_dist > 0 else 1000
            distance_change_pct = distance_change / current_distance if current_distance > 0 else 0

            # Distance comfort (similar to last race)
            distance_comfort = distance_change <= 200  # Within 200m is comfortable

            # Exact distance match
            exact_distance_match = distance_change <= 50

            # Distance comfort score
            if exact_distance_match:
                comfort_score = 1.0
            elif distance_comfort:
                comfort_score = 0.5
            elif distance_change_pct < 0.20:  # Within 20% change
                comfort_score = 0.0
            else:
                comfort_score = -0.5  # Significant distance change

            distance_analysis[idx] = {
                'current_distance': current_distance,
                'last_distance': last_dist,
                'distance_change': distance_change,
                'distance_change_pct': distance_change_pct,
                'distance_comfort': distance_comfort,
                'exact_distance_match': exact_distance_match,
                'comfort_score': comfort_score
            }

        return distance_analysis

    def _calculate_realistic_competitive_scores(self, earnings_analysis: Dict, track_analysis: Dict,
                                              form_analysis: Dict, connection_analysis: Dict,
                                              distance_analysis: Dict) -> Dict[str, Any]:
        """
        Calculate data-realistic competitive scores using only reliable fields.

        Args:
            earnings_analysis: Earnings quality analysis
            track_analysis: Track specialization analysis
            form_analysis: Form momentum analysis
            connection_analysis: Hot connection analysis
            distance_analysis: Distance comfort analysis

        Returns:
            Composite competitive scores for each horse
        """
        competitive_scores = {}

        # Get all horse indices
        all_indices = set()
        for analysis in [earnings_analysis, track_analysis, form_analysis, connection_analysis, distance_analysis]:
            all_indices.update(analysis.keys())

        if self.verbose:
            self.logger.info(f"=== REALISTIC COMPETITIVE SCORING ===")

        for idx in all_indices:
            # Extract individual advantages
            earnings_data = earnings_analysis.get(idx, {})
            track_data = track_analysis.get(idx, {})
            form_data = form_analysis.get(idx, {})
            connection_data = connection_analysis.get(idx, {})
            distance_data = distance_analysis.get(idx, {})

            # Individual advantage scores
            earnings_score = earnings_data.get('earnings_score', 0.0)
            track_score = track_data.get('track_score', 0.0)
            form_score = form_data.get('form_score', 0.0)
            connection_score = connection_data.get('connection_score', 0.0)
            distance_score = distance_data.get('comfort_score', 0.0)

            # Individual advantages detection
            advantages = {
                'earnings_advantage': earnings_data.get('class_advantage', False),
                'track_specialist': track_data.get('track_specialist', False),
                'positive_momentum': form_data.get('positive_momentum', False),
                'hot_connection': connection_data.get('hot_connection', False),
                'distance_comfort': distance_data.get('distance_comfort', False)
            }

            # Count total advantages
            total_advantages = sum(advantages.values())

            # Calculate weighted composite score (conservative weighting)
            composite_score = (
                earnings_score * 0.25 +     # Earnings quality (class advantage)
                track_score * 0.25 +        # Track specialization
                form_score * 0.25 +         # Form momentum
                connection_score * 0.15 +   # Hot connections
                distance_score * 0.10       # Distance comfort
            )

            # Determine primary advantage type
            advantage_scores = {
                'earnings': earnings_score,
                'track': track_score,
                'form': form_score,
                'connection': connection_score,
                'distance': distance_score
            }

            if total_advantages > 1:
                primary_advantage = 'multiple'
                advantage_strength = composite_score
            elif total_advantages == 1:
                primary_advantage = max(advantage_scores.items(), key=lambda x: x[1])[0]
                advantage_strength = max(advantage_scores.values())
            else:
                primary_advantage = 'none'
                advantage_strength = 0.0

            # Categorize competitive strength
            competitive_strength = self._categorize_realistic_competitive_strength(
                composite_score, total_advantages
            )

            competitive_scores[idx] = {
                'earnings_score': earnings_score,
                'track_score': track_score,
                'form_score': form_score,
                'connection_score': connection_score,
                'distance_score': distance_score,
                'composite_score': composite_score,
                'advantages': advantages,
                'total_advantages': total_advantages,
                'primary_advantage_type': primary_advantage,
                'advantage_strength': advantage_strength,
                'competitive_strength': competitive_strength
            }

        return competitive_scores

    def _categorize_realistic_competitive_strength(self, composite_score: float,
                                                 total_advantages: int) -> str:
        """
        Categorize competitive strength based on realistic data analysis.

        Args:
            composite_score: Weighted composite competitive score
            total_advantages: Number of individual category advantages

        Returns:
            Competitive strength category
        """
        if total_advantages >= 3 and composite_score > 0.3:
            return 'dominant'
        elif total_advantages >= 2 and composite_score > 0.15:
            return 'strong'
        elif total_advantages >= 1 or composite_score > 0.05:
            return 'moderate'
        elif composite_score > -0.1:
            return 'neutral'
        else:
            return 'weak'

    def _apply_realistic_competitive_adjustments(self, base_predictions: np.ndarray,
                                               competitive_scores: Dict[str, Any],
                                               race_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply conservative competitive adjustments based on realistic data analysis.

        Args:
            base_predictions: Base model predictions array
            competitive_scores: Realistic competitive analysis scores
            race_data: Race participant data

        Returns:
            Tuple of (adjusted_predictions, adjustment_details)
        """
        if base_predictions is None:
            raise ValueError("base_predictions cannot be None")

        adjusted_predictions = base_predictions.copy()
        adjustment_details = {}

        # Conservative adjustment scaling
        adjustment_scaling = {
            'earnings': 0.25,      # Max ±25% for earnings advantage
            'track': 0.20,         # Max ±20% for track specialist
            'form': 0.15,          # Max ±15% for form momentum
            'connection': 0.10,    # Max ±10% for hot connections
            'distance': 0.05       # Max ±5% for distance comfort
        }

        for i, idx in enumerate(race_data.index):
            if idx not in competitive_scores:
                adjustment_details[idx] = {'total_adjustment': 0.0, 'components': {}}
                continue

            scores = competitive_scores[idx]

            # Calculate individual adjustments based on advantages
            advantages = scores['advantages']
            adjustment_components = {}
            total_adjustment = 0.0

            # Earnings advantage adjustment
            if advantages.get('earnings_advantage', False):
                earnings_adj = -adjustment_scaling['earnings']  # Negative = better position
                adjustment_components['earnings_adjustment'] = earnings_adj
                total_adjustment += earnings_adj

            # Track specialist adjustment
            if advantages.get('track_specialist', False):
                track_adj = -adjustment_scaling['track']
                adjustment_components['track_adjustment'] = track_adj
                total_adjustment += track_adj

            # Form momentum adjustment
            if advantages.get('positive_momentum', False):
                form_adj = -adjustment_scaling['form']
                adjustment_components['form_adjustment'] = form_adj
                total_adjustment += form_adj

            # Hot connection adjustment
            if advantages.get('hot_connection', False):
                connection_adj = -adjustment_scaling['connection']
                adjustment_components['connection_adjustment'] = connection_adj
                total_adjustment += connection_adj

            # Distance comfort adjustment
            if advantages.get('distance_comfort', False):
                distance_adj = -adjustment_scaling['distance']
                adjustment_components['distance_adjustment'] = distance_adj
                total_adjustment += distance_adj

            # Cap maximum total adjustment at ±0.5 positions
            total_adjustment = np.clip(total_adjustment, -0.5, 0.5)

            # Apply adjustment as percentage change to prediction
            original_prediction = base_predictions[i]
            adjustment_factor = 1.0 + total_adjustment
            adjusted_predictions[i] = original_prediction * adjustment_factor

            # Store adjustment details
            adjustment_details[idx] = {
                'original_prediction': original_prediction,
                'total_adjustment': total_adjustment,
                'adjustment_factor': adjustment_factor,
                'final_prediction': adjusted_predictions[i],
                'components': adjustment_components,
                'primary_advantage_type': scores['primary_advantage_type'],
                'advantage_strength': scores['advantage_strength'],
                'competitive_strength': scores['competitive_strength']
            }

        return adjusted_predictions, adjustment_details

    def _create_realistic_competitive_analysis(self, race_data: pd.DataFrame,
                                             earnings_analysis: Dict, track_analysis: Dict,
                                             form_analysis: Dict, connection_analysis: Dict,
                                             distance_analysis: Dict, competitive_scores: Dict) -> Dict[str, Any]:
        """
        Create comprehensive realistic competitive analysis summary.

        Args:
            race_data: Race participant data
            earnings_analysis: Earnings quality analysis
            track_analysis: Track specialization analysis
            form_analysis: Form momentum analysis
            connection_analysis: Hot connection analysis
            distance_analysis: Distance comfort analysis
            competitive_scores: Composite competitive scores

        Returns:
            Comprehensive realistic competitive analysis
        """
        analysis_summary = {
            'field_size': len(race_data),
            'analysis_categories': {
                'earnings_quality': earnings_analysis,
                'track_specialization': track_analysis,
                'form_momentum': form_analysis,
                'hot_connections': connection_analysis,
                'distance_comfort': distance_analysis
            },
            'competitive_scores': competitive_scores,
            'field_statistics': self._calculate_realistic_field_statistics(competitive_scores),
            'top_contenders': self._identify_realistic_top_contenders(competitive_scores, top_n=3),
            'advantage_summary': self._summarize_field_advantages(competitive_scores),
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'data_realistic',
                'reliable_data_only': True,
                'conservative_adjustments': True
            }
        }

        return analysis_summary

    def _calculate_realistic_field_statistics(self, competitive_scores: Dict) -> Dict[str, Any]:
        """Calculate statistical summary of realistic field competitive strength."""
        if not competitive_scores:
            return {}

        scores = [data['composite_score'] for data in competitive_scores.values()]
        advantages = [data['total_advantages'] for data in competitive_scores.values()]

        return {
            'average_competitive_score': np.mean(scores),
            'std_competitive_score': np.std(scores),
            'max_competitive_score': np.max(scores),
            'min_competitive_score': np.min(scores),
            'average_advantages': np.mean(advantages),
            'horses_with_advantages': len([a for a in advantages if a > 0]),
            'field_competitiveness': 'high' if np.std(scores) > 0.15 else 'moderate' if np.std(scores) > 0.08 else 'low'
        }

    def _identify_realistic_top_contenders(self, competitive_scores: Dict, top_n: int = 3) -> List[Dict]:
        """Identify top realistic competitive contenders."""
        sorted_horses = sorted(
            competitive_scores.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )

        return [
            {
                'horse_index': idx,
                'composite_score': data['composite_score'],
                'competitive_strength': data['competitive_strength'],
                'total_advantages': data['total_advantages'],
                'primary_advantage_type': data['primary_advantage_type'],
                'advantage_strength': data['advantage_strength'],
                'key_advantages': [k for k, v in data['advantages'].items() if v]
            }
            for idx, data in sorted_horses[:top_n]
        ]

    def _summarize_field_advantages(self, competitive_scores: Dict) -> Dict[str, Any]:
        """Summarize the distribution of advantages across the field."""
        if not competitive_scores:
            return {}

        advantage_counts = {
            'earnings_advantage': 0,
            'track_specialist': 0,
            'positive_momentum': 0,
            'hot_connection': 0,
            'distance_comfort': 0
        }

        for data in competitive_scores.values():
            for advantage, has_advantage in data['advantages'].items():
                if has_advantage:
                    advantage_counts[advantage] += 1

        total_horses = len(competitive_scores)
        return {
            'advantage_counts': advantage_counts,
            'advantage_percentages': {k: (v/total_horses)*100 for k, v in advantage_counts.items()},
            'total_horses': total_horses,
            'horses_with_multiple_advantages': sum(1 for data in competitive_scores.values() if data['total_advantages'] > 1)
        }

    def _calculate_competitive_scores(self, speed_analysis: Dict, track_analysis: Dict,
                                    class_analysis: Dict, form_analysis: Dict) -> Dict[str, Any]:
        """
        Calculate composite competitive scores for each horse (legacy method).

        Args:
            speed_analysis: Speed dominance analysis results
            track_analysis: Track specialization analysis results
            class_analysis: Class relief analysis results
            form_analysis: Form momentum analysis results

        Returns:
            Composite competitive scores for each horse
        """
        competitive_scores = {}

        # Get all horse indices
        all_indices = set(speed_analysis.keys()) | set(track_analysis.keys()) | \
                     set(class_analysis.keys()) | set(form_analysis.keys())

        for idx in all_indices:
            # Extract individual scores
            speed_score = speed_analysis.get(idx, {}).get('speed_score', 0.0)
            track_score = track_analysis.get(idx, {}).get('track_score', 0.0)
            class_score = class_analysis.get(idx, {}).get('class_score', 0.0)
            form_score = form_analysis.get(idx, {}).get('form_score', 0.0)

            # Calculate weighted composite score
            composite_score = (
                speed_score * self.analysis_weights['speed_dominance'] +
                track_score * self.analysis_weights['track_specialization'] +
                class_score * self.analysis_weights['class_relief'] +
                form_score * self.analysis_weights['form_momentum']
            )

            # Individual category advantages (for detailed analysis)
            advantages = {
                'speed_advantage': speed_score > self.thresholds['speed_dominance_pct'],
                'track_advantage': track_score > self.thresholds['track_win_rate_diff'],
                'class_advantage': class_score > self.thresholds['class_drop_threshold'],
                'form_advantage': form_score > self.thresholds['form_trend_threshold']
            }

            # Count total advantages
            total_advantages = sum(advantages.values())

            competitive_scores[idx] = {
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

    def _categorize_competitive_strength(self, composite_score: float,
                                       total_advantages: int) -> str:
        """
        Categorize competitive strength based on composite score and advantages.

        Args:
            composite_score: Weighted composite competitive score
            total_advantages: Number of individual category advantages

        Returns:
            Competitive strength category
        """
        if composite_score > 0.3 and total_advantages >= 3:
            return 'dominant'
        elif composite_score > 0.15 and total_advantages >= 2:
            return 'strong'
        elif composite_score > 0.05 or total_advantages >= 1:
            return 'moderate'
        elif composite_score > -0.1:
            return 'neutral'
        else:
            return 'weak'

    def _apply_competitive_adjustments(self, base_predictions: np.ndarray,
                                     competitive_scores: Dict[str, Any],
                                     race_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply competitive adjustments to base model predictions.

        Args:
            base_predictions: Base model predictions array
            competitive_scores: Competitive analysis scores
            race_data: Race participant data

        Returns:
            Tuple of (adjusted_predictions, adjustment_details)
        """
        if base_predictions is None:
            raise ValueError("base_predictions cannot be None")

        adjusted_predictions = base_predictions.copy()
        adjustment_details = {}

        for i, idx in enumerate(race_data.index):
            if idx not in competitive_scores:
                adjustment_details[idx] = {'total_adjustment': 0.0, 'components': {}}
                continue

            scores = competitive_scores[idx]

            # Calculate individual adjustments
            speed_adj = scores['speed_score'] * self.adjustment_scaling['speed_dominance']
            track_adj = scores['track_score'] * self.adjustment_scaling['track_specialization']
            class_adj = scores['class_score'] * self.adjustment_scaling['class_relief']
            form_adj = scores['form_score'] * self.adjustment_scaling['form_momentum']

            # Total adjustment (sum of components)
            total_adjustment = speed_adj + track_adj + class_adj + form_adj

            # Apply adjustment as percentage change
            original_prediction = base_predictions[i]
            adjustment_factor = 1.0 + total_adjustment
            adjusted_predictions[i] = original_prediction * adjustment_factor

            # Store adjustment details
            adjustment_details[idx] = {
                'original_prediction': original_prediction,
                'total_adjustment': total_adjustment,
                'adjustment_factor': adjustment_factor,
                'final_prediction': adjusted_predictions[i],
                'components': {
                    'speed_adjustment': speed_adj,
                    'track_adjustment': track_adj,
                    'class_adjustment': class_adj,
                    'form_adjustment': form_adj
                },
                'competitive_strength': scores['competitive_strength']
            }

        return adjusted_predictions, adjustment_details

    def _create_competitive_analysis(self, race_data: pd.DataFrame,
                                   speed_analysis: Dict, track_analysis: Dict,
                                   class_analysis: Dict, form_analysis: Dict,
                                   competitive_scores: Dict) -> Dict[str, Any]:
        """
        Create comprehensive competitive analysis summary.

        Args:
            race_data: Race participant data
            speed_analysis: Speed dominance analysis
            track_analysis: Track specialization analysis
            class_analysis: Class relief analysis
            form_analysis: Form momentum analysis
            competitive_scores: Composite competitive scores

        Returns:
            Comprehensive competitive analysis
        """
        analysis_summary = {
            'field_size': len(race_data),
            'analysis_categories': {
                'speed_dominance': speed_analysis,
                'track_specialization': track_analysis,
                'class_relief': class_analysis,
                'form_momentum': form_analysis
            },
            'competitive_scores': competitive_scores,
            'field_statistics': self._calculate_field_statistics(competitive_scores),
            'top_contenders': self._identify_top_contenders(competitive_scores, top_n=3),
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'analysis_weights': self.analysis_weights,
                'adjustment_scaling': self.adjustment_scaling,
                'thresholds': self.thresholds
            }
        }

        return analysis_summary

    def _calculate_field_statistics(self, competitive_scores: Dict) -> Dict[str, Any]:
        """Calculate statistical summary of field competitive strength."""
        if not competitive_scores:
            return {}

        scores = [data['composite_score'] for data in competitive_scores.values()]
        advantages = [data['total_advantages'] for data in competitive_scores.values()]

        return {
            'average_competitive_score': np.mean(scores),
            'std_competitive_score': np.std(scores),
            'max_competitive_score': np.max(scores),
            'min_competitive_score': np.min(scores),
            'average_advantages': np.mean(advantages),
            'horses_with_advantages': len([a for a in advantages if a > 0]),
            'field_competitiveness': 'high' if np.std(scores) > 0.2 else 'moderate' if np.std(scores) > 0.1 else 'low'
        }

    def _identify_top_contenders(self, competitive_scores: Dict, top_n: int = 3) -> List[Dict]:
        """Identify top competitive contenders."""
        sorted_horses = sorted(
            competitive_scores.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )

        return [
            {
                'horse_index': idx,
                'composite_score': data['composite_score'],
                'competitive_strength': data['competitive_strength'],
                'total_advantages': data['total_advantages'],
                'key_advantages': [k for k, v in data['advantages'].items() if v]
            }
            for idx, data in sorted_horses[:top_n]
        ]

    def _create_adjustment_summary(self, adjustment_details: Dict) -> Dict[str, Any]:
        """Create summary of competitive adjustments applied."""
        if not adjustment_details:
            return {}

        # Handle nested structure where adjustment_details contains model -> horse_idx -> adjustment data
        all_adjustments = []
        for model_name, model_adjustments in adjustment_details.items():
            if isinstance(model_adjustments, dict):
                for horse_idx, horse_data in model_adjustments.items():
                    if isinstance(horse_data, dict) and 'total_adjustment' in horse_data:
                        all_adjustments.append(horse_data['total_adjustment'])

        # If no adjustments found, try flat structure
        if not all_adjustments:
            for data in adjustment_details.values():
                if isinstance(data, dict) and 'total_adjustment' in data:
                    all_adjustments.append(data['total_adjustment'])

        if not all_adjustments:
            return {
                'total_horses_adjusted': 0,
                'horses_boosted': 0,
                'horses_penalized': 0,
                'average_adjustment': 0.0,
                'max_boost': 0.0,
                'max_penalty': 0.0,
                'adjustment_spread': 0.0,
                'significant_adjustments': 0
            }

        positive_adjustments = [adj for adj in all_adjustments if adj > 0]
        negative_adjustments = [adj for adj in all_adjustments if adj < 0]

        return {
            'total_horses_adjusted': len([adj for adj in all_adjustments if abs(adj) > 0.001]),
            'horses_boosted': len(positive_adjustments),
            'horses_penalized': len(negative_adjustments),
            'average_adjustment': np.mean(all_adjustments),
            'max_boost': max(positive_adjustments) if positive_adjustments else 0.0,
            'max_penalty': min(negative_adjustments) if negative_adjustments else 0.0,
            'adjustment_spread': np.std(all_adjustments),
            'significant_adjustments': len([adj for adj in all_adjustments if abs(adj) > 0.05])
        }

    def _generate_audit_trail(self, race_data: pd.DataFrame,
                            base_predictions: Dict[str, np.ndarray],
                            enhanced_predictions: Dict[str, np.ndarray],
                            competitive_analysis: Dict[str, Any],
                            adjustment_details: Dict[str, Any],
                            race_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive audit trail for competitive analysis.

        Args:
            race_data: Race participant data
            base_predictions: Original model predictions
            enhanced_predictions: Competitively adjusted predictions
            competitive_analysis: Detailed competitive analysis
            adjustment_details: Adjustment details per model
            race_metadata: Race context information

        Returns:
            Complete audit trail
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'race_metadata': race_metadata,
            'field_size': len(race_data),
            'models_processed': list(base_predictions.keys()),
            'base_predictions': {
                model: preds.tolist() for model, preds in base_predictions.items()
            },
            'enhanced_predictions': {
                model: preds.tolist() for model, preds in enhanced_predictions.items()
            },
            'competitive_analysis_summary': {
                'field_statistics': competitive_analysis.get('field_statistics', {}),
                'top_contenders': competitive_analysis.get('top_contenders', []),
                'analysis_metadata': competitive_analysis.get('analysis_metadata', {})
            },
            'adjustment_summary': {
                model: self._create_adjustment_summary(details)
                for model, details in adjustment_details.items()
            },
            'performance_expectations': self._estimate_performance_impact(
                base_predictions, enhanced_predictions, competitive_analysis
            )
        }

    def _estimate_performance_impact(self, base_predictions: Dict[str, np.ndarray],
                                   enhanced_predictions: Dict[str, np.ndarray],
                                   competitive_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate expected performance impact of competitive adjustments.

        Args:
            base_predictions: Original predictions
            enhanced_predictions: Adjusted predictions
            competitive_analysis: Competitive analysis results

        Returns:
            Performance impact estimation
        """
        impact_estimates = {}

        for model_name in base_predictions.keys():
            base_preds = base_predictions[model_name]
            enhanced_preds = enhanced_predictions[model_name]

            # Calculate prediction changes
            prediction_changes = enhanced_preds - base_preds
            avg_change = np.mean(np.abs(prediction_changes))
            max_change = np.max(np.abs(prediction_changes))

            # Estimate R² improvement based on competitive strength
            field_stats = competitive_analysis.get('field_statistics', {})
            competitiveness = field_stats.get('field_competitiveness', 'moderate')

            # Conservative improvement estimates
            if competitiveness == 'high' and avg_change > 0.05:
                estimated_r2_improvement = 0.015  # +1.5% R²
            elif competitiveness == 'moderate' and avg_change > 0.03:
                estimated_r2_improvement = 0.010  # +1.0% R²
            else:
                estimated_r2_improvement = 0.005  # +0.5% R²

            impact_estimates[model_name] = {
                'average_prediction_change': avg_change,
                'max_prediction_change': max_change,
                'estimated_r2_improvement': estimated_r2_improvement,
                'confidence_level': 'high' if competitiveness == 'high' else 'moderate'
            }

        return impact_estimates

    def save_analysis_results(self, analysis_results: Dict[str, Any],
                            output_path: str) -> str:
        """
        Save competitive analysis results to file.

        Args:
            analysis_results: Complete analysis results
            output_path: Output file path

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(analysis_results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        if self.verbose:
            self.logger.info(f"Competitive analysis results saved to {output_path}")

        return str(output_path)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj