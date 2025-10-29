"""
Quinte-Specific Feature Calculator

This module calculates specialized features for Quinté+ races by analyzing
historical quinte performance from the historical_quinte table.

Features are designed to capture quinte-specific patterns that differ from
regular races due to larger field sizes, higher purses, and unique betting dynamics.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from core.calculators.static_feature_calculator import FeatureCalculator


class QuinteFeatureCalculator:
    """
    Calculate quinte-specific features for horses, races, and post positions.

    Features are calculated from historical_quinte table data, focusing on:
    - Horse-level quinte career statistics
    - Race-level quinte characteristics
    - Post position biases in quinte races
    """

    def __init__(self, db_path: str):
        """
        Initialize the calculator with database connection.

        Args:
            db_path: Path to SQLite database containing historical_quinte table
        """
        self.db_path = db_path
        self._batch_data_cache = None

    @staticmethod
    def _safe_numeric(value, default=0.0):
        """Safely convert value to numeric."""
        return FeatureCalculator.safe_numeric(value, default)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely divide two numbers, returning default if denominator is 0."""
        if denominator == 0 or not np.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if np.isfinite(result) else default

    def get_horse_quinte_history(self, horse_id: str, before_date: str = None) -> List[Dict]:
        """
        Get all quinte races for a specific horse from historical_quinte.

        Args:
            horse_id: Horse identifier (idche)
            before_date: Only get races before this date (for temporal consistency)

        Returns:
            List of race dictionaries with participant data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        date_filter = f"AND jour < '{before_date}'" if before_date else ""

        query = f"""
        SELECT comp, jour, participants, dist, partant, typec, handi_raw,
               is_handicap, hippo, natpis
        FROM historical_quinte
        WHERE 1=1 {date_filter}
        ORDER BY jour DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        # Parse results
        horse_races = []
        for row in rows:
            comp, jour, participants_json, dist, partant, typec, handi_raw, is_handicap, hippo, natpis = row

            # Parse participants JSON
            try:
                participants = json.loads(participants_json)

                # Find this horse in participants
                for participant in participants:
                    if str(participant.get('idche')) == str(horse_id):
                        race_data = {
                            'comp': comp,
                            'jour': jour,
                            'dist': dist,
                            'partant': partant,
                            'typec': typec,
                            'handi_raw': handi_raw,
                            'is_handicap': is_handicap,
                            'hippo': hippo,
                            'natpis': natpis,
                            'participant': participant
                        }
                        horse_races.append(race_data)
                        break
            except (json.JSONDecodeError, TypeError):
                continue

        return horse_races

    def get_race_results(self, race_comp: str) -> Optional[Dict[int, int]]:
        """
        Get race results from quinte_results table.

        Args:
            race_comp: Race identifier

        Returns:
            Dict mapping horse_numero to finish_position, or None if not found
            Example: {7: 1, 16: 2, 1: 3, ...}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT ordre_arrivee
                FROM quinte_results
                WHERE comp = ?
            """, (race_comp,))

            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                # Parse ordre_arrivee as JSON array
                # Format: [{"narrivee": "1", "cheval": 7, "idche": 1248485}, ...]
                results_data = json.loads(row[0])

                # Create mapping of horse_numero -> finish_position
                horse_positions = {}
                for result in results_data:
                    narrivee = result.get('narrivee', '')
                    cheval = result.get('cheval', 0)

                    # Only include numeric positions (skip "DAI", "ARR", etc.)
                    if str(narrivee).isdigit() and cheval > 0:
                        position = int(narrivee)
                        horse_positions[int(cheval)] = position

                return horse_positions if horse_positions else None

            return None

        except Exception:
            conn.close()
            return None

    def calculate_horse_quinte_features(self, horse_id: str, current_race_info: Dict,
                                       before_date: str = None) -> Dict[str, float]:
        """
        Calculate horse-level quinté features.

        Features:
        - quinte_career_starts: Total Quinté races
        - quinte_win_rate: Win % in Quinté
        - quinte_top5_rate: Top 5 % in Quinté
        - avg_quinte_position: Average finish in Quinté
        - days_since_last_quinte: Recency in Quinté
        - quinte_handicap_specialist: Better in handicaps (0/1)
        - quinte_conditions_specialist: Better in conditions (0/1)
        - quinte_large_field_ability: Performance in 15+ fields
        - quinte_track_condition_fit: Performance in today's condition

        Args:
            horse_id: Horse identifier
            current_race_info: Dict with current race details (dist, typec, natpis, etc.)
            before_date: Only use races before this date

        Returns:
            Dict of calculated features
        """
        features = {}

        # Get horse's quinte history
        quinte_races = self.get_horse_quinte_history(horse_id, before_date)

        # Basic stats
        quinte_starts = len(quinte_races)
        features['quinte_career_starts'] = float(quinte_starts)

        if quinte_starts == 0:
            # No quinte history - return defaults
            return {
                'quinte_career_starts': 0.0,
                'quinte_win_rate': 0.0,
                'quinte_top5_rate': 0.0,
                'avg_quinte_position': 10.0,  # Neutral position
                'days_since_last_quinte': 999.0,  # Very long time
                'quinte_handicap_specialist': 0.0,
                'quinte_conditions_specialist': 0.0,
                'quinte_large_field_ability': 0.5,  # Neutral
                'quinte_track_condition_fit': 0.5,  # Neutral
            }

        # Analyze performance
        wins = 0
        top5_finishes = 0
        positions = []
        handicap_positions = []
        conditions_positions = []
        large_field_positions = []  # 15+ starters
        current_condition_positions = []

        most_recent_date = None

        for race in quinte_races:
            race_comp = race['comp']
            participant = race['participant']

            # Get results (now returns dict of horse_numero -> position)
            horse_positions = self.get_race_results(race_comp)

            if horse_positions:
                # Find horse's finish position
                horse_numero = int(self._safe_numeric(participant.get('numero'), 0))

                if horse_numero in horse_positions:
                    position = horse_positions[horse_numero]  # Direct lookup
                    positions.append(position)

                    # Count wins and top 5
                    if position == 1:
                        wins += 1
                    if position <= 5:
                        top5_finishes += 1

                    # Specialist analysis
                    is_handicap = race.get('is_handicap', 0)
                    if is_handicap:
                        handicap_positions.append(position)
                    else:
                        conditions_positions.append(position)

                    # Large field ability
                    field_size = self._safe_numeric(race.get('partant'), 0)
                    if field_size >= 15:
                        large_field_positions.append(position)

                    # Track condition fit
                    race_condition = str(race.get('natpis', '')).strip().upper()
                    current_condition = str(current_race_info.get('natpis', '')).strip().upper()
                    if race_condition == current_condition:
                        current_condition_positions.append(position)

            # Track most recent race date
            race_date = race.get('jour')
            if race_date:
                if most_recent_date is None or race_date > most_recent_date:
                    most_recent_date = race_date

        # Calculate win rate and top 5 rate
        races_with_results = len(positions)
        features['quinte_win_rate'] = self._safe_divide(wins, races_with_results, 0.0)
        features['quinte_top5_rate'] = self._safe_divide(top5_finishes, races_with_results, 0.0)

        # Average position
        if positions:
            features['avg_quinte_position'] = float(np.mean(positions))
        else:
            features['avg_quinte_position'] = 10.0  # Neutral

        # Days since last quinte
        if most_recent_date:
            try:
                if before_date:
                    current_date = datetime.strptime(before_date, '%Y-%m-%d')
                else:
                    current_date = datetime.now()

                last_quinte_date = datetime.strptime(most_recent_date, '%Y-%m-%d')
                days_since = (current_date - last_quinte_date).days
                features['days_since_last_quinte'] = min(float(days_since), 999.0)
            except (ValueError, TypeError):
                features['days_since_last_quinte'] = 999.0
        else:
            features['days_since_last_quinte'] = 999.0

        # Specialist indicators (better performance in specific race types)
        avg_handicap_pos = np.mean(handicap_positions) if handicap_positions else 10.0
        avg_conditions_pos = np.mean(conditions_positions) if conditions_positions else 10.0

        # Consider specialist if avg position is 2+ places better
        features['quinte_handicap_specialist'] = 1.0 if (avg_handicap_pos < avg_conditions_pos - 2.0) else 0.0
        features['quinte_conditions_specialist'] = 1.0 if (avg_conditions_pos < avg_handicap_pos - 2.0) else 0.0

        # Large field ability (normalized performance)
        if large_field_positions:
            avg_large_field_pos = np.mean(large_field_positions)
            # Normalize: position 1-5 is good (0.8-1.0), 10+ is poor (0.0-0.2)
            features['quinte_large_field_ability'] = max(0.0, min(1.0, 1.0 - (avg_large_field_pos - 1) / 15.0))
        else:
            features['quinte_large_field_ability'] = 0.5  # Neutral

        # Track condition fit
        if current_condition_positions:
            avg_condition_pos = np.mean(current_condition_positions)
            # Normalize: better positions = higher score
            features['quinte_track_condition_fit'] = max(0.0, min(1.0, 1.0 - (avg_condition_pos - 1) / 15.0))
        else:
            features['quinte_track_condition_fit'] = 0.5  # Neutral

        return features

    def calculate_race_quinte_features(self, race_info: Dict) -> Dict[str, float]:
        """
        Calculate race-level quinté features.

        Features:
        - is_handicap_quinte: Handicap (1) or Conditions (0)
        - handicap_division: 0, 1, or 2
        - purse_level_category: Low/Medium/High (0/1/2)
        - field_size_category: 14-15 / 16-17 / 18+ (0/1/2)
        - track_condition: PH/DUR/PS/PSF encoded
        - weather_conditions: From meteo field

        Args:
            race_info: Dict with race details (is_handicap, handicap_division, cheque,
                      partant, natpis, meteo, etc.)

        Returns:
            Dict of calculated features
        """
        features = {}

        # Handicap vs Conditions
        features['is_handicap_quinte'] = float(race_info.get('is_handicap', 0))

        # Handicap division
        handicap_division = self._safe_numeric(race_info.get('handicap_division', 0), 0.0)
        features['handicap_division'] = min(handicap_division, 2.0)  # Cap at 2

        # Purse level category
        purse = self._safe_numeric(race_info.get('cheque', 0), 0.0)
        if purse < 30000:
            features['purse_level_category'] = 0.0  # Low
        elif purse < 100000:
            features['purse_level_category'] = 1.0  # Medium
        else:
            features['purse_level_category'] = 2.0  # High

        # Field size category
        field_size = self._safe_numeric(race_info.get('partant'), 15)
        if field_size <= 15:
            features['field_size_category'] = 0.0  # 14-15
        elif field_size <= 17:
            features['field_size_category'] = 1.0  # 16-17
        else:
            features['field_size_category'] = 2.0  # 18+

        # Track condition (one-hot encoding)
        natpis = str(race_info.get('natpis', '')).strip().upper()
        features['track_condition_PH'] = 1.0 if natpis == 'PH' else 0.0  # Heavy
        features['track_condition_DUR'] = 1.0 if natpis == 'DUR' else 0.0  # Hard
        features['track_condition_PS'] = 1.0 if natpis == 'PS' else 0.0  # Soft
        features['track_condition_PSF'] = 1.0 if natpis == 'PSF' else 0.0  # All-weather

        # Weather conditions (simplified encoding)
        meteo = str(race_info.get('meteo', '')).strip().upper()
        features['weather_clear'] = 1.0 if 'ENSOLEILLE' in meteo or 'CLAIR' in meteo else 0.0
        features['weather_rain'] = 1.0 if 'PLUIE' in meteo or 'PLUVIEUX' in meteo else 0.0
        features['weather_cloudy'] = 1.0 if 'NUAGEUX' in meteo or 'COUVERT' in meteo else 0.0

        return features

    def calculate_post_position_features(self, post_position: int, race_info: Dict) -> Dict[str, float]:
        """
        Calculate post position quinté features.

        Features:
        - post_position_bias: Historical advantage at this gate (from all quinte races)
        - post_position_track_bias: Gate bias at this specific track/condition

        Args:
            post_position: Gate number (1-20)
            race_info: Dict with race details (hippo, natpis)

        Returns:
            Dict of calculated features
        """
        features = {}

        # Get historical post position performance
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Overall post position bias
        cursor.execute("""
            SELECT comp, participants
            FROM historical_quinte
            LIMIT 1000
        """)

        all_races = cursor.fetchall()

        position_wins = {i: 0 for i in range(1, 21)}
        position_starts = {i: 0 for i in range(1, 21)}
        position_top5 = {i: 0 for i in range(1, 21)}

        track_position_wins = {i: 0 for i in range(1, 21)}
        track_position_starts = {i: 0 for i in range(1, 21)}

        current_track = race_info.get('hippo', '')
        current_condition = race_info.get('natpis', '')

        for race_comp, participants_json in all_races:
            try:
                participants = json.loads(participants_json)
                results = self.get_race_results(race_comp)

                if not results:
                    continue

                for participant in participants:
                    numero = int(self._safe_numeric(participant.get('numero'), 0))

                    if numero > 0 and numero <= 20:
                        position_starts[numero] += 1

                        # Direct lookup from dict
                        if numero in results:
                            finish_pos = results[numero]  # Already the position

                            if finish_pos == 1:
                                position_wins[numero] += 1
                            if finish_pos <= 5:
                                position_top5[numero] += 1

                # Track-specific stats (would need race info from historical_quinte)
                # Simplified: just count overall for now

            except (json.JSONDecodeError, TypeError):
                continue

        conn.close()

        # Calculate bias for this post position
        if post_position in position_starts and position_starts[post_position] > 0:
            win_rate = self._safe_divide(position_wins[post_position], position_starts[post_position], 0.05)
            top5_rate = self._safe_divide(position_top5[post_position], position_starts[post_position], 0.3)

            # Normalize: expected win rate is ~1/16 = 6.25% for quinte
            # Expected top 5 rate is 5/16 = 31.25%
            features['post_position_bias'] = (win_rate / 0.0625 + top5_rate / 0.3125) / 2.0
        else:
            features['post_position_bias'] = 1.0  # Neutral

        # Track-specific bias (simplified for now - would need more sophisticated analysis)
        features['post_position_track_bias'] = 1.0  # Neutral default

        return features

    def calculate_all_quinte_features(self, horse_id: str, race_info: Dict,
                                     post_position: int, before_date: str = None) -> Dict[str, float]:
        """
        Calculate all quinte-specific features for a horse in a race.

        Args:
            horse_id: Horse identifier
            race_info: Dict with race details
            post_position: Gate number
            before_date: Only use historical data before this date

        Returns:
            Dict with all quinte features combined
        """
        features = {}

        # Horse-level features
        horse_features = self.calculate_horse_quinte_features(horse_id, race_info, before_date)
        features.update(horse_features)

        # Race-level features
        race_features = self.calculate_race_quinte_features(race_info)
        features.update(race_features)

        # Post position features
        post_features = self.calculate_post_position_features(post_position, race_info)
        features.update(post_features)

        return features

    def batch_load_all_quinte_data(self) -> Dict:
        """
        Batch-load ALL quinté historical data in a single pass.

        This loads all races and results from the database once, avoiding
        thousands of individual queries.

        Returns:
            Dict containing:
            - 'races': List of all race dictionaries with parsed participants
            - 'results': Dict mapping race_comp to horse_positions {numero: position}
            - 'horse_index': Dict mapping horse_id to list of race indices
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load all quinté races
        cursor.execute("""
            SELECT comp, jour, participants, dist, partant, typec, handi_raw,
                   is_handicap, hippo, natpis, cheque, handicap_division, meteo
            FROM historical_quinte
            ORDER BY jour DESC
        """)

        races = []
        horse_index = {}  # Maps horse_id -> list of race indices

        for idx, row in enumerate(cursor.fetchall()):
            comp, jour, participants_json, dist, partant, typec, handi_raw, is_handicap, hippo, natpis, cheque, handicap_division, meteo = row

            try:
                participants = json.loads(participants_json)

                race_data = {
                    'comp': comp,
                    'jour': jour,
                    'dist': dist,
                    'partant': partant,
                    'typec': typec,
                    'handi_raw': handi_raw,
                    'is_handicap': is_handicap,
                    'hippo': hippo,
                    'natpis': natpis,
                    'cheque': cheque,
                    'handicap_division': handicap_division,
                    'meteo': meteo,
                    'participants': participants
                }
                races.append(race_data)

                # Build horse index
                for participant in participants:
                    horse_id = str(participant.get('idche', ''))
                    if horse_id:
                        if horse_id not in horse_index:
                            horse_index[horse_id] = []
                        horse_index[horse_id].append(idx)

            except (json.JSONDecodeError, TypeError):
                continue

        # Load all quinté results
        cursor.execute("SELECT comp, ordre_arrivee FROM quinte_results")

        results = {}
        for comp, ordre_arrivee in cursor.fetchall():
            try:
                results_data = json.loads(ordre_arrivee)
                horse_positions = {}

                for result in results_data:
                    narrivee = result.get('narrivee', '')
                    cheval = result.get('cheval', 0)

                    if str(narrivee).isdigit() and cheval > 0:
                        position = int(narrivee)
                        horse_positions[int(cheval)] = position

                if horse_positions:
                    results[comp] = horse_positions

            except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
                continue

        conn.close()

        return {
            'races': races,
            'results': results,
            'horse_index': horse_index
        }

    def add_batch_quinte_features(self, df: pd.DataFrame, race_info: Dict,
                                  before_date: str, all_data: Dict) -> pd.DataFrame:
        """
        Add quinté features to a DataFrame using pre-loaded batch data.

        This is much faster than querying the database for each horse individually.

        Args:
            df: DataFrame with participant data
            race_info: Dict with race details
            before_date: Only use historical data before this date
            all_data: Pre-loaded data from batch_load_all_quinte_data()

        Returns:
            DataFrame with quinté features added
        """
        result_df = df.copy()

        # Initialize all feature columns with default values
        default_features = {
            'quinte_career_starts': 0.0,
            'quinte_win_rate': 0.0,
            'quinte_top5_rate': 0.0,
            'avg_quinte_position': 10.0,
            'days_since_last_quinte': 999.0,
            'quinte_handicap_specialist': 0.0,
            'quinte_conditions_specialist': 0.0,
            'quinte_large_field_ability': 0.5,
            'quinte_track_condition_fit': 0.5,
        }

        for feature_name, default_value in default_features.items():
            result_df[feature_name] = default_value

        # Calculate features for each horse
        for index, row in result_df.iterrows():
            horse_id = str(row.get('idche', ''))
            post_position = int(self._safe_numeric(row.get('numero'), 1))

            if horse_id and horse_id != 'dummy':
                # Get horse-level features from batch data
                horse_features = self._batch_calculate_horse_features(
                    horse_id, race_info, before_date, all_data
                )

                for feature_name, feature_value in horse_features.items():
                    result_df.at[index, feature_name] = feature_value

        # Add race-level features (same for all horses in the race)
        race_features = self.calculate_race_quinte_features(race_info)
        for feature_name, feature_value in race_features.items():
            result_df[feature_name] = feature_value

        # Add post position features
        for index, row in result_df.iterrows():
            post_position = int(self._safe_numeric(row.get('numero'), 1))
            post_features = self._batch_calculate_post_features(
                post_position, race_info, all_data
            )

            for feature_name, feature_value in post_features.items():
                result_df.at[index, feature_name] = feature_value

        return result_df

    def _batch_calculate_horse_features(self, horse_id: str, current_race_info: Dict,
                                       before_date: str, all_data: Dict) -> Dict[str, float]:
        """
        Calculate horse-level quinté features from batch-loaded data.

        Args:
            horse_id: Horse identifier
            current_race_info: Dict with current race details
            before_date: Only use races before this date
            all_data: Pre-loaded data from batch_load_all_quinte_data()

        Returns:
            Dict of calculated features
        """
        features = {}

        # Get this horse's race history from the index
        horse_index = all_data['horse_index']
        results = all_data['results']
        races = all_data['races']

        if horse_id not in horse_index:
            # No history - return defaults
            return {
                'quinte_career_starts': 0.0,
                'quinte_win_rate': 0.0,
                'quinte_top5_rate': 0.0,
                'avg_quinte_position': 10.0,
                'days_since_last_quinte': 999.0,
                'quinte_handicap_specialist': 0.0,
                'quinte_conditions_specialist': 0.0,
                'quinte_large_field_ability': 0.5,
                'quinte_track_condition_fit': 0.5,
            }

        # Get horse's races (filtered by date)
        race_indices = horse_index[horse_id]
        horse_races = []

        for race_idx in race_indices:
            race = races[race_idx]

            # Skip races after the current date
            if before_date and race['jour'] >= before_date:
                continue

            # Find this horse in the participants
            for participant in race['participants']:
                if str(participant.get('idche')) == horse_id:
                    horse_races.append({
                        'race': race,
                        'participant': participant
                    })
                    break

        # Calculate statistics
        quinte_starts = len(horse_races)
        features['quinte_career_starts'] = float(quinte_starts)

        if quinte_starts == 0:
            return {
                'quinte_career_starts': 0.0,
                'quinte_win_rate': 0.0,
                'quinte_top5_rate': 0.0,
                'avg_quinte_position': 10.0,
                'days_since_last_quinte': 999.0,
                'quinte_handicap_specialist': 0.0,
                'quinte_conditions_specialist': 0.0,
                'quinte_large_field_ability': 0.5,
                'quinte_track_condition_fit': 0.5,
            }

        # Analyze performance
        wins = 0
        top5_finishes = 0
        positions = []
        handicap_positions = []
        conditions_positions = []
        large_field_positions = []
        current_condition_positions = []
        most_recent_date = None

        for horse_race_data in horse_races:
            race = horse_race_data['race']
            participant = horse_race_data['participant']
            race_comp = race['comp']

            # Get results
            if race_comp in results:
                horse_numero = int(self._safe_numeric(participant.get('numero'), 0))

                if horse_numero in results[race_comp]:
                    position = results[race_comp][horse_numero]
                    positions.append(position)

                    if position == 1:
                        wins += 1
                    if position <= 5:
                        top5_finishes += 1

                    # Specialist analysis
                    if race.get('is_handicap', 0):
                        handicap_positions.append(position)
                    else:
                        conditions_positions.append(position)

                    # Large field ability
                    field_size = self._safe_numeric(race.get('partant'), 0)
                    if field_size >= 15:
                        large_field_positions.append(position)

                    # Track condition fit
                    race_condition = str(race.get('natpis', '')).strip().upper()
                    current_condition = str(current_race_info.get('natpis', '')).strip().upper()
                    if race_condition == current_condition:
                        current_condition_positions.append(position)

            # Track most recent race date
            race_date = race.get('jour')
            if race_date:
                if most_recent_date is None or race_date > most_recent_date:
                    most_recent_date = race_date

        # Calculate win rate and top 5 rate
        races_with_results = len(positions)
        features['quinte_win_rate'] = self._safe_divide(wins, races_with_results, 0.0)
        features['quinte_top5_rate'] = self._safe_divide(top5_finishes, races_with_results, 0.0)

        # Average position
        features['avg_quinte_position'] = float(np.mean(positions)) if positions else 10.0

        # Days since last quinte
        if most_recent_date:
            try:
                if before_date:
                    current_date = datetime.strptime(before_date, '%Y-%m-%d')
                else:
                    current_date = datetime.now()

                last_quinte_date = datetime.strptime(most_recent_date, '%Y-%m-%d')
                days_since = (current_date - last_quinte_date).days
                features['days_since_last_quinte'] = min(float(days_since), 999.0)
            except (ValueError, TypeError):
                features['days_since_last_quinte'] = 999.0
        else:
            features['days_since_last_quinte'] = 999.0

        # Specialist indicators
        avg_handicap_pos = np.mean(handicap_positions) if handicap_positions else 10.0
        avg_conditions_pos = np.mean(conditions_positions) if conditions_positions else 10.0

        features['quinte_handicap_specialist'] = 1.0 if (avg_handicap_pos < avg_conditions_pos - 2.0) else 0.0
        features['quinte_conditions_specialist'] = 1.0 if (avg_conditions_pos < avg_handicap_pos - 2.0) else 0.0

        # Large field ability
        if large_field_positions:
            avg_large_field_pos = np.mean(large_field_positions)
            features['quinte_large_field_ability'] = max(0.0, min(1.0, 1.0 - (avg_large_field_pos - 1) / 15.0))
        else:
            features['quinte_large_field_ability'] = 0.5

        # Track condition fit
        if current_condition_positions:
            avg_condition_pos = np.mean(current_condition_positions)
            features['quinte_track_condition_fit'] = max(0.0, min(1.0, 1.0 - (avg_condition_pos - 1) / 15.0))
        else:
            features['quinte_track_condition_fit'] = 0.5

        return features

    def _batch_calculate_post_features(self, post_position: int, race_info: Dict,
                                      all_data: Dict) -> Dict[str, float]:
        """
        Calculate post position features from batch-loaded data.

        Args:
            post_position: Gate number (1-20)
            race_info: Dict with race details
            all_data: Pre-loaded data from batch_load_all_quinte_data()

        Returns:
            Dict of calculated features
        """
        features = {}

        races = all_data['races']
        results = all_data['results']

        position_wins = {i: 0 for i in range(1, 21)}
        position_starts = {i: 0 for i in range(1, 21)}
        position_top5 = {i: 0 for i in range(1, 21)}

        # Use first 1000 races for post position analysis
        for race in races[:1000]:
            race_comp = race['comp']

            if race_comp not in results:
                continue

            race_results = results[race_comp]

            for participant in race['participants']:
                numero = int(self._safe_numeric(participant.get('numero'), 0))

                if numero > 0 and numero <= 20:
                    position_starts[numero] += 1

                    if numero in race_results:
                        finish_pos = race_results[numero]

                        if finish_pos == 1:
                            position_wins[numero] += 1
                        if finish_pos <= 5:
                            position_top5[numero] += 1

        # Calculate bias for this post position
        if post_position in position_starts and position_starts[post_position] > 0:
            win_rate = self._safe_divide(position_wins[post_position], position_starts[post_position], 0.05)
            top5_rate = self._safe_divide(position_top5[post_position], position_starts[post_position], 0.3)

            # Normalize: expected win rate is ~1/16 = 6.25% for quinte
            # Expected top 5 rate is 5/16 = 31.25%
            features['post_position_bias'] = (win_rate / 0.0625 + top5_rate / 0.3125) / 2.0
        else:
            features['post_position_bias'] = 1.0  # Neutral

        # Track-specific bias (simplified)
        features['post_position_track_bias'] = 1.0  # Neutral default

        return features


def add_quinte_features_to_dataframe(df: pd.DataFrame, db_path: str,
                                    race_info: Dict, before_date: str = None) -> pd.DataFrame:
    """
    Add quinte-specific features to a DataFrame of race participants.

    Args:
        df: DataFrame with participant data (must have 'idche' and 'numero' columns)
        db_path: Path to SQLite database
        race_info: Dict with race details
        before_date: Only use historical data before this date

    Returns:
        DataFrame with quinte features added
    """
    calculator = QuinteFeatureCalculator(db_path)
    result_df = df.copy()

    # Initialize all feature columns
    sample_features = calculator.calculate_all_quinte_features(
        horse_id='dummy',
        race_info=race_info,
        post_position=1,
        before_date=before_date
    )

    for feature_name in sample_features.keys():
        result_df[feature_name] = 0.0

    # Calculate features for each horse
    for index, row in result_df.iterrows():
        horse_id = str(row.get('idche', ''))
        post_position = int(FeatureCalculator.safe_numeric(row.get('numero'), 1))

        if horse_id and horse_id != 'dummy':
            features = calculator.calculate_all_quinte_features(
                horse_id=horse_id,
                race_info=race_info,
                post_position=post_position,
                before_date=before_date
            )

            for feature_name, feature_value in features.items():
                result_df.at[index, feature_name] = feature_value

    return result_df
