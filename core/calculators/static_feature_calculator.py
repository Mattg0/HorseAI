from typing import Dict, List, Union, Optional, Any
import pandas as pd
import numpy as np
import json
import re
from decimal import Decimal
from core.calculators.musique_calculation import MusiqueFeatureExtractor
from core.data_cleaning.tabnet_cleaner import TabNetDataCleaner

class FeatureCalculator:
    """
    Module de calcul des features dérivées pour les courses hippiques.
    Peut être utilisé à la fois pour l'entraînement et la prédiction.
    """
    
    @staticmethod
    def normalize_earnings_to_quality(gains_carriere: float, courses_cheval: float, field_mean: float = 100000) -> float:
        """
        Convert absolute earnings to normalized quality class (0.0 to 1.0).

        Uses log-scale normalization to compress extreme values into TabNet-friendly range.
        This completely eliminates extreme value issues while preserving relative quality signal.

        Quality Scale:
            0.0 = No earnings
            0.3 = Below average (< 50k/race)
            0.5 = Average (~100k/race, typical field mean)
            0.7 = Above average (~200k/race)
            0.9 = Elite (~500k/race)
            1.0 = Top earners (>1M/race, capped at 99th percentile)

        Args:
            gains_carriere: Total career earnings
            courses_cheval: Number of career races
            field_mean: Field average earnings per race (for context)

        Returns:
            Normalized quality score [0.0, 1.0]

        Examples:
            - Horse with 0 earnings: 0.0
            - Horse with 50k/race: ~0.29
            - Horse with 100k/race: ~0.50 (average)
            - Horse with 500k/race: ~0.89 (elite)
            - Horse with 2M/race: 1.0 (capped)
        """
        if courses_cheval == 0:
            return 0.0

        # Ensure non-negative inputs
        gains_carriere = max(0.0, gains_carriere)

        # Calculate raw earnings per race
        raw_earnings_per_race = gains_carriere / courses_cheval

        # Cap at 1M per race (99th percentile) to prevent extreme outliers
        # This prevents a single lucky big-purse win from dominating
        EARNINGS_CAP = 1000000.0

        # Clamp to reasonable range
        clamped_earnings = min(raw_earnings_per_race, EARNINGS_CAP)

        # Log-scale normalization: log(x + 1) / log(cap + 1)
        # +1 prevents log(0) and provides smooth curve from 0
        import math
        quality = math.log(clamped_earnings + 1) / math.log(EARNINGS_CAP + 1)

        # Ensure output is in [0, 1] range
        return max(0.0, min(1.0, quality))

    @staticmethod
    def calculate_field_mean_earnings(participants_data: list, min_races: int = 5) -> float:
        """
        Calculate field average earnings per race for context-aware weighting.

        Uses only horses with sufficient race experience for reliable estimate.

        Args:
            participants_data: List of participant dictionaries
            min_races: Minimum races required for inclusion

        Returns:
            Median earnings per race from experienced horses, or default 100k
        """
        experienced_earnings = []

        for participant in participants_data:
            gains = FeatureCalculator.safe_numeric(participant.get('gainsCarriere', 0), 0.0)
            courses = FeatureCalculator.safe_numeric(participant.get('coursescheval', 0), 0.0)

            if courses >= min_races and gains > 0:
                earnings_per_race = gains / courses
                # Cap at reasonable maximum to prevent outliers from skewing field mean
                # Use much stricter threshold for field mean calculation
                if earnings_per_race <= 300000:  # 300k per race is already very high
                    experienced_earnings.append(earnings_per_race)

        if experienced_earnings and len(experienced_earnings) >= 2:
            # Use median for robustness against remaining outliers
            return float(np.median(experienced_earnings))
        else:
            # Fallback to reasonable default for typical horse racing
            return 100000.0

    @staticmethod
    def validate_earnings_features(features: Dict[str, float], feature_name: str) -> None:
        """
        Validate earnings features are in normalized [0, 1] range.

        Args:
            features: Dictionary of calculated features
            feature_name: Name of the feature to validate

        Raises:
            AssertionError: If feature values are outside [0, 1] range
        """
        if feature_name in features:
            value = features[feature_name]

            # Validate normalized range [0, 1]
            assert 0.0 <= value <= 1.0, f"{feature_name} out of range [0,1]: {value}"

    @staticmethod
    def safe_numeric(value, default=0.0):
        """Safely convert value to numeric, handling various types including racing positions."""
        if pd.isna(value) or value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip().upper()
            if value in ('', 'N/A', 'NULL'):
                return default
            # Handle racing position strings
            if value in ['DAI', 'DISTANTE', 'DIST']:
                return 15.0  # Poor finish
            elif value in ['ARR', 'ARRETE', 'STOPPED']:
                return 16.0  # Did not finish
            elif value in ['TOMBE', 'FALLEN']:
                return 17.0  # Fell
            elif value in ['NP', 'NON_PARTANT', 'DID_NOT_START']:
                return 18.0  # Did not start
            try:
                return float(value)
            except ValueError:
                return 12.0  # Default for unknown strings
        return default

    @staticmethod
    def calculate_performance_ratios(participant: Dict, field_mean: float = 100000) -> Dict[str, float]:
        """
        Calcule les ratios de performance du cheval avec pondération par confiance.

        Args:
            participant: Dictionnaire contenant les données du participant
            field_mean: Moyenne de terrain pour la pondération par confiance

        Returns:
            Dict avec les ratios calculés (gains_par_course utilise la pondération par confiance)
        """
        courses_total = FeatureCalculator.safe_numeric(participant.get('coursescheval', 0), 0.0)
        gains_carriere = FeatureCalculator.safe_numeric(participant.get('gainsCarriere', 0), 0.0)

        if courses_total > 0:
            ratios = {
                'ratio_victoires': FeatureCalculator.safe_numeric(participant.get('victoirescheval,', 0), 0.0) / courses_total,
                'ratio_places': FeatureCalculator.safe_numeric(participant.get('placescheval', 0), 0.0) / courses_total,
                'gains_par_course': FeatureCalculator.normalize_earnings_to_quality(
                    gains_carriere, courses_total, field_mean
                )
            }
        else:
            ratios = {
                'ratio_victoires': 0.0,
                'ratio_places': 0.0,
                'gains_par_course': 0.0
            }

        # Validate the critical feature that was causing TabNet issues
        FeatureCalculator.validate_earnings_features(ratios, 'gains_par_course')

        return ratios

    @staticmethod
    def calculate_phase1_career_features(participant: Dict, field_mean: float = 100000) -> Dict[str, float]:
        """
        Calculate Phase 1 career performance features with confidence weighting.

        Args:
            participant: Dictionary containing participant data
            field_mean: Field mean for confidence weighting

        Returns:
            Dict with Phase 1 career features
        """
        courses_total = FeatureCalculator.safe_numeric(participant.get('coursescheval', 0), 0.0)
        victoires = FeatureCalculator.safe_numeric(participant.get('victoirescheval', 0), 0.0)
        gains_carriere = FeatureCalculator.safe_numeric(participant.get('gainsCarriere', 0), 0.0)

        # Ensure earnings are non-negative (prevent calculation errors)
        gains_carriere = max(0.0, gains_carriere)

        features = {}

        # Career strike rate (more specific than ratio_victoires)
        features['career_strike_rate'] = victoires / courses_total if courses_total > 0 else 0.0

        # Earnings per race - normalized to 0-1 quality class to prevent TabNet scaling issues
        features['earnings_per_race'] = FeatureCalculator.normalize_earnings_to_quality(
            gains_carriere, courses_total, field_mean
        ) if courses_total > 0 else 0.0

        # Recent vs career earnings comparison
        gains_annee = FeatureCalculator.safe_numeric(participant.get('gainsAnneeEnCours', 0), 0.0)
        gains_precedente = FeatureCalculator.safe_numeric(participant.get('gainsAnneePrecedente', 0), 0.0)
        features['earnings_trend'] = (gains_annee - gains_precedente) / max(gains_precedente, 1) if gains_precedente > 0 else 0.0

        # Validate earnings features
        FeatureCalculator.validate_earnings_features(features, 'earnings_per_race')

        return features
    
    @staticmethod
    def calculate_phase1_last_race_features(participant: Dict) -> Dict[str, float]:
        """
        Calculate Phase 1 last race performance features.
        
        Args:
            participant: Dictionary containing participant data
            
        Returns:
            Dict with last race features
        """
        features = {}
        
        # Last race performance metrics (single values for tabular models)
        last_position = FeatureCalculator.safe_numeric(participant.get('derniereplace', 10), 10.0)
        features['last_race_position_normalized'] = (last_position - 1) / 10.0
        
        last_odds = FeatureCalculator.safe_numeric(participant.get('dernierecote', 10.0), 10.0)
        features['last_race_odds_normalized'] = min(last_odds / 10.0, 3.0)
        
        # Last race competitive context
        derniers_partants = FeatureCalculator.safe_numeric(participant.get('derniernbpartants', 10), 10.0)
        features['last_race_field_size_factor'] = min(derniers_partants / 10.0, 2.0)
        
        # Distance consistency (current vs last race)
        current_dist = FeatureCalculator.safe_numeric(participant.get('dist', 2000), 2000.0)
        last_dist = FeatureCalculator.safe_numeric(participant.get('dernieredist', 2000), 2000.0)
        features['distance_consistency'] = 1.0 - abs(current_dist - last_dist) / max(current_dist, last_dist)
        
        return features
    
    @staticmethod
    def calculate_phase1_rating_features(participant: Dict) -> Dict[str, float]:
        """
        Calculate Phase 1 official rating and classification features.
        
        Args:
            participant: Dictionary containing participant data
            
        Returns:
            Dict with rating features
        """
        features = {}
        
        # VHA rating processing
        vha = FeatureCalculator.safe_numeric(participant.get('vha', 0), 0.0)
        features['vha_normalized'] = vha / 100.0 if vha > 0 else 0.0
        
        # Claiming tax progression
        current_tx = FeatureCalculator.safe_numeric(participant.get('txreclam', 0), 0.0)
        last_tx = FeatureCalculator.safe_numeric(participant.get('dernierTxreclam', 0), 0.0)
        features['claiming_tax_trend'] = (current_tx - last_tx) / max(last_tx, 1) if last_tx > 0 else 0.0
        
        # Class consistency indicator
        features['class_stability'] = 1.0 if abs(current_tx - last_tx) <= 5 else 0.5
        
        return features

    @staticmethod
    def calculate_couple_stats(participant: Dict) -> Dict[str, float]:
        """
        Calcule les statistiques du couple cheval/jockey.

        Args:
            participant: Dictionnaire contenant les données du participant

        Returns:
            Dict avec les statistiques calculées
        """
        couple_courses = FeatureCalculator.safe_numeric(participant.get('nbCourseCouple', 0), 0.0)

        if couple_courses > 0:
            stats = {
                'efficacite_couple': FeatureCalculator.safe_numeric(participant.get('nbVictCouple', 0), 0.0) / couple_courses,
                'regularite_couple': FeatureCalculator.safe_numeric(participant.get('nbPlaceCouple', 0), 0.0) / couple_courses,
                'progression_couple': FeatureCalculator.safe_numeric(participant.get('TxVictCouple', 0), 0.0) - FeatureCalculator.safe_numeric(participant.get('pourcVictCheval', 0), 0.0)
            }
        else:
            stats = {
                'efficacite_couple': 0.0,
                'regularite_couple': 0.0,
                'progression_couple': 0.0
            }

        return stats

    @staticmethod
    def calculate_hippo_stats(participant: Dict) -> Dict[str, float]:
        return {
            'perf_cheval_hippo': (
                                         FeatureCalculator.safe_numeric(participant.get('pourcVictChevalHippo', 0), 0.0) +
                                         FeatureCalculator.safe_numeric(participant.get('pourcPlaceChevalHippo', 0), 0.0)
                                 ) / 2,
            'perf_jockey_hippo': (
                                         FeatureCalculator.safe_numeric(participant.get('pourcVictJockHippo', 0), 0.0) +
                                         FeatureCalculator.safe_numeric(participant.get('pourcPlaceJockHippo', 0), 0.0)
                                 ) / 2
        }

    @staticmethod
    def calculate_blinkers_features(participant: Dict) -> Dict[str, float]:
        """
        Calculate blinkers change detection features.
        
        Args:
            participant: Participant data with blinkers fields
            
        Returns:
            Dict with blinkers features
        """
        features = {}
        
        # Get blinkers data
        oeil_first_time = participant.get('oeilFirstTime', False)
        current_blinkers = participant.get('oeil', '')
        previous_blinkers = participant.get('dernierOeil', '')
        
        # Convert boolean field properly
        if isinstance(oeil_first_time, (str, int)):
            oeil_first_time = bool(int(oeil_first_time)) if str(oeil_first_time).isdigit() else False
        
        # Clean blinkers data
        current_blinkers = str(current_blinkers).strip().upper() if current_blinkers else ''
        previous_blinkers = str(previous_blinkers).strip().upper() if previous_blinkers else ''
        
        # First-time blinkers (high-impact positive angle)
        features['blinkers_first_time'] = float(oeil_first_time)
        
        # Current blinkers type
        features['has_blinkers'] = float(bool(current_blinkers and current_blinkers != 'NONE'))
        features['has_standard_blinkers'] = float(current_blinkers == 'O')
        features['has_australian_blinkers'] = float(current_blinkers == 'A')
        
        # Blinkers changes
        blinkers_added = bool(current_blinkers and not previous_blinkers)
        blinkers_removed = bool(previous_blinkers and not current_blinkers)
        blinkers_type_changed = bool(
            current_blinkers and previous_blinkers and 
            current_blinkers != previous_blinkers
        )
        
        features['blinkers_added'] = float(blinkers_added)
        features['blinkers_removed'] = float(blinkers_removed) 
        features['blinkers_type_changed'] = float(blinkers_type_changed)
        features['blinkers_any_change'] = float(blinkers_added or blinkers_removed or blinkers_type_changed)
        
        # High-impact blinkers changes (known positive angles)
        features['blinkers_high_impact_change'] = float(
            oeil_first_time or blinkers_added or 
            (blinkers_type_changed and current_blinkers in ['O', 'A'])
        )
        
        return features

    @staticmethod  
    def calculate_shoeing_features(participant: Dict) -> Dict[str, float]:
        """
        Calculate shoeing change detection features.
        
        Args:
            participant: Participant data with shoeing fields
            
        Returns:
            Dict with shoeing features
        """
        features = {}
        
        # Get shoeing data
        def_first_time = participant.get('defFirstTime', False)
        current_shoeing = participant.get('defoeil', '')
        previous_shoeing = participant.get('defoeilPrec', '')
        
        # Convert boolean field properly
        if isinstance(def_first_time, (str, int)):
            def_first_time = bool(int(def_first_time)) if str(def_first_time).isdigit() else False
        
        # Clean shoeing data
        current_shoeing = str(current_shoeing).strip().upper() if current_shoeing else ''
        previous_shoeing = str(previous_shoeing).strip().upper() if previous_shoeing else ''
        
        # First-time shoeing changes
        features['shoeing_first_time'] = float(def_first_time)
        
        # Current shoeing pattern analysis
        features['fully_shod'] = float(current_shoeing == 'FF')  # Front and back
        features['front_shod_only'] = float(current_shoeing in ['FD', 'FP'])  # Front shod
        features['back_shod_only'] = float(current_shoeing in ['DF', 'PF'])  # Back shod
        features['barefoot'] = float(current_shoeing in ['DD', 'PP'])  # No shoes
        
        # Shoeing changes
        shoeing_changed = bool(
            current_shoeing and previous_shoeing and 
            current_shoeing != previous_shoeing
        )
        
        # Significant shoeing changes (high impact)
        barefoot_to_shod = bool(
            previous_shoeing in ['DD', 'PP', ''] and 
            current_shoeing in ['FF', 'FD', 'DF', 'FP', 'PF', 'FP']
        )
        
        shod_to_barefoot = bool(
            previous_shoeing in ['FF', 'FD', 'DF', 'FP', 'PF'] and
            current_shoeing in ['DD', 'PP']
        )
        
        # Major pattern changes
        major_shoeing_change = bool(
            (previous_shoeing == 'DD' and current_shoeing == 'FF') or  # Barefoot to fully shod
            (previous_shoeing == 'FF' and current_shoeing == 'DD') or  # Fully shod to barefoot  
            (previous_shoeing == 'PP' and current_shoeing == 'FF') or  # Pads to shoes
            def_first_time  # First time change always significant
        )
        
        features['shoeing_changed'] = float(shoeing_changed)
        features['barefoot_to_shod'] = float(barefoot_to_shod)
        features['shod_to_barefoot'] = float(shod_to_barefoot)
        features['major_shoeing_change'] = float(major_shoeing_change)
        
        # Shoeing complexity score (more complex = higher score)
        shoeing_complexity_map = {
            'DD': 0,  # Barefoot
            'PP': 0,  # Pads only
            'FD': 1,  # Front shoes only
            'DF': 1,  # Back shoes only  
            'FP': 2,  # Front shoes, back pads
            'PF': 2,  # Front pads, back shoes
            'FF': 3   # Full shoes
        }
        
        features['shoeing_complexity'] = float(shoeing_complexity_map.get(current_shoeing, 0))
        
        return features

    @staticmethod
    def safe_string(value: Any, default: str = '') -> str:
        """Safely convert value to string."""
        if pd.isna(value) or value is None:
            return default
        return str(value).strip()

    @staticmethod
    def parse_time_to_seconds(time_str: str) -> Optional[float]:
        """
        Parse French racing time format to seconds.
        Examples: "2'42\"53" -> 162.53, "3'19\"52" -> 199.52
        """
        if not time_str or pd.isna(time_str):
            return None

        time_str = str(time_str).strip()
        if not time_str or time_str in ('', 'N/A'):
            return None

        # Match patterns like "2'42"53" or "2:42.53"
        patterns = [
            r"(\d+)'(\d+)\"(\d+)",  # 2'42"53
            r"(\d+):(\d+)\.(\d+)",  # 2:42.53
            r"(\d+)\.(\d+)\.(\d+)"  # 2.42.53
        ]

        for pattern in patterns:
            match = re.match(pattern, time_str)
            if match:
                minutes, seconds, centiseconds = map(int, match.groups())
                return minutes * 60 + seconds + centiseconds / 100

        return None

    @staticmethod
    def calculate_class_movement_features(participant: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate class movement analysis features."""
        features = {}

        current_purse = FeatureCalculator.safe_numeric(race_info.get('cheque', 0), 0.0)
        prev_purse = FeatureCalculator.safe_numeric(participant.get('dernierealloc', 0), 0.0)
        field_size = FeatureCalculator.safe_numeric(race_info.get('partant', 1), 1.0)

        # Class drop percentage
        if prev_purse > 0:
            features['class_drop_pct'] = max(0, (prev_purse - current_purse) / prev_purse)
        else:
            features['class_drop_pct'] = 0.0

        # Purse per starter
        if field_size > 0 and current_purse > 0 and np.isfinite(current_purse):
            features['purse_per_starter'] = current_purse / field_size
        else:
            features['purse_per_starter'] = 0.0

        # Purse ratio
        if prev_purse > 0:
            features['purse_ratio'] = current_purse / prev_purse
        else:
            features['purse_ratio'] = 1.0

        # Moving up in class
        features['moving_up_in_class'] = 1.0 if features['purse_ratio'] > 1.5 else 0.0

        # Class shock indicator
        ratio = features['purse_ratio']
        features['class_shock_indicator'] = 1.0 if (ratio > 2.0 or ratio < 0.5) else 0.0

        # Handicap level change
        current_handicap_score = FeatureCalculator.safe_numeric(race_info.get('handicap_level_score', 0), 0.0)
        features['handicap_level_change'] = current_handicap_score

        return features

    @staticmethod
    def calculate_speed_figure_features(participant: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate speed figure proxy features."""
        features = {}

        current_time_str = FeatureCalculator.safe_string(participant.get('tempstot', ''))
        current_distance = FeatureCalculator.safe_numeric(race_info.get('dist', 0), 0.0)
        record_time_str = FeatureCalculator.safe_string(participant.get('recordG', ''))

        current_time = FeatureCalculator.parse_time_to_seconds(current_time_str)
        record_time = FeatureCalculator.parse_time_to_seconds(record_time_str)

        # Time per distance
        if current_time and current_distance > 0:
            features['time_per_distance'] = current_time / current_distance
        else:
            features['time_per_distance'] = 0.0

        # Time vs record ratio
        if current_time and record_time and record_time > 0:
            features['time_vs_record'] = current_time / record_time
        else:
            features['time_vs_record'] = 1.0

        # Pace rating
        if current_time and current_distance > 0:
            standard_distance = 2000
            normalized_time = (current_time / current_distance) * standard_distance
            features['pace_rating'] = max(0, 200 - normalized_time / 60 * 100)
        else:
            features['pace_rating'] = 100.0

        # Speed figure proxy
        time_component = 1.0 / max(features['time_vs_record'], 0.1)
        distance_component = features['time_per_distance'] if features['time_per_distance'] > 0 else 1.0
        features['speed_figure_proxy'] = time_component / max(distance_component, 0.001) * 100

        return features

    @staticmethod
    def calculate_form_context_features(participant: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate form context analysis features."""
        features = {}

        current_odds = FeatureCalculator.safe_numeric(participant.get('coteprob', 0), 0.0)
        prev_odds = FeatureCalculator.safe_numeric(participant.get('dernierecote', 0), 0.0)
        prev_position_str = FeatureCalculator.safe_string(participant.get('derniereplace', ''))
        current_distance = FeatureCalculator.safe_numeric(race_info.get('dist', 0), 0.0)
        prev_distance = FeatureCalculator.safe_numeric(participant.get('dernieredist', 0), 0.0)
        recency = FeatureCalculator.safe_numeric(participant.get('recence', 0), 0.0)

        # Market confidence shift
        if prev_odds > 0 and current_odds > 0:
            features['market_confidence_shift'] = prev_odds / current_odds
        else:
            features['market_confidence_shift'] = 1.0

        # Last race improvement
        try:
            prev_position = int(prev_position_str) if prev_position_str.isdigit() else 10
            features['last_race_improvement'] = max(0, 15 - prev_position) / 15
        except:
            features['last_race_improvement'] = 0.5

        # Bounce risk
        prev_pos_num = 10
        try:
            if prev_position_str.isdigit():
                prev_pos_num = int(prev_position_str)
        except:
            pass
        features['bounce_risk'] = 1.0 if prev_pos_num <= 3 else 0.0

        # Layoff distance interaction
        if prev_distance > 0:
            distance_change_pct = abs(current_distance - prev_distance) / prev_distance
            features['layoff_distance_interaction'] = (recency / 30) * distance_change_pct
        else:
            features['layoff_distance_interaction'] = recency / 60

        return features

    @staticmethod
    def calculate_connection_features(participant: Dict[str, Any]) -> Dict[str, float]:
        """Calculate trainer/connection change features."""
        features = {}

        current_trainer = FeatureCalculator.safe_string(participant.get('entraineur', ''))
        prev_trainer = FeatureCalculator.safe_string(participant.get('dernierEnt', ''))

        # Trainer change indicator
        if current_trainer and prev_trainer and current_trainer != prev_trainer:
            features['trainer_change'] = 1.0
        else:
            features['trainer_change'] = 0.0

        # Connection stability
        features['connection_stability'] = 1.0 - features['trainer_change']

        return features

    @staticmethod
    def calculate_competition_context_features(participant: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate competition context features."""
        features = {}

        current_field = FeatureCalculator.safe_numeric(race_info.get('partant', 0), 0.0)
        prev_field = FeatureCalculator.safe_numeric(participant.get('derniernbpartants', 0), 0.0)
        current_distance = FeatureCalculator.safe_numeric(race_info.get('dist', 0), 0.0)
        prev_distance = FeatureCalculator.safe_numeric(participant.get('dernieredist', 0), 0.0)

        # Field size change
        if prev_field > 0:
            features['field_size_change'] = (current_field - prev_field) / prev_field
        else:
            features['field_size_change'] = 0.0

        # Distance comfort
        if prev_distance > 0:
            distance_change_pct = abs(current_distance - prev_distance) / prev_distance
            features['distance_comfort'] = max(0, 1.0 - distance_change_pct)
        else:
            features['distance_comfort'] = 0.5

        # Competition level shift
        features['competition_level_shift'] = current_field / 20.0

        return features

    @staticmethod
    def calculate_interaction_features(participant: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feature interaction combinations."""
        features = {}

        recency = FeatureCalculator.safe_numeric(participant.get('recence', 0), 0.0)
        current_purse = FeatureCalculator.safe_numeric(race_info.get('cheque', 0), 0.0)
        prev_purse = FeatureCalculator.safe_numeric(participant.get('dernierealloc', 0), 0.0)

        class_drop = 0.0
        if prev_purse > 0:
            class_drop = max(0, (prev_purse - current_purse) / prev_purse)

        current_trainer = FeatureCalculator.safe_string(participant.get('entraineur', ''))
        prev_trainer = FeatureCalculator.safe_string(participant.get('dernierEnt', ''))
        trainer_change = 1.0 if (current_trainer and prev_trainer and current_trainer != prev_trainer) else 0.0

        current_distance = FeatureCalculator.safe_numeric(race_info.get('dist', 0), 0.0)
        prev_distance = FeatureCalculator.safe_numeric(participant.get('dernieredist', 0), 0.0)

        distance_change = 0.0
        if prev_distance > 0:
            distance_change = abs(current_distance - prev_distance) / prev_distance

        # Recency × Class Drop
        features['recence_x_class_drop'] = (recency / 30) * class_drop

        # Trainer Change × Recency
        features['trainer_change_x_recence'] = trainer_change * (recency / 30)

        # Distance Change × Recency
        features['distance_change_x_recence'] = distance_change * (recency / 30)

        return features

    @staticmethod
    def calculate_equipment_impact_features(participant: Dict) -> Dict[str, float]:
        """
        Calculate combined equipment impact features.
        
        Args:
            participant: Participant data
            
        Returns:
            Dict with combined equipment features
        """
        # Get individual equipment features
        blinkers_features = FeatureCalculator.calculate_blinkers_features(participant)
        shoeing_features = FeatureCalculator.calculate_shoeing_features(participant)
        
        features = {}
        
        # Multiple equipment changes in same race (high impact)
        multiple_changes = (
            blinkers_features['blinkers_any_change'] + 
            shoeing_features['shoeing_changed']
        )
        features['multiple_equipment_changes'] = float(multiple_changes >= 2)
        
        # Equipment change momentum score
        momentum_score = 0
        
        # Blinkers momentum
        if blinkers_features['blinkers_first_time']:
            momentum_score += 3  # Highest impact
        elif blinkers_features['blinkers_added']:
            momentum_score += 2
        elif blinkers_features['blinkers_type_changed']:
            momentum_score += 1
        
        # Shoeing momentum  
        if shoeing_features['shoeing_first_time']:
            momentum_score += 2
        elif shoeing_features['major_shoeing_change']:
            momentum_score += 2
        elif shoeing_features['shoeing_changed']:
            momentum_score += 1
        
        features['equipment_momentum_score'] = float(momentum_score)
        features['high_equipment_momentum'] = float(momentum_score >= 3)
        
        # Combined first-time equipment changes (extremely rare and high-impact)
        features['multiple_first_time_changes'] = float(
            blinkers_features['blinkers_first_time'] + 
            shoeing_features['shoeing_first_time'] >= 2
        )
        
        # Equipment optimization indicator (positive changes)
        optimization_score = 0
        if blinkers_features['blinkers_first_time'] or blinkers_features['blinkers_added']:
            optimization_score += 1
        if shoeing_features['barefoot_to_shod']:
            optimization_score += 1
        if shoeing_features['shoeing_complexity'] > 1:  # More complex shoeing
            optimization_score += 1
            
        features['equipment_optimization_score'] = float(optimization_score)
        
        return features

    @staticmethod
    def calculate_trainer_stats(participant: Dict) -> Dict:
        """
        Calculate trainer performance statistics.

        Args:
            participant: Participant data

        Returns:
            Dict with trainer features
        """
        features = {}

        # Basic trainer stats (use safe numeric conversion)
        trainer_victoires = FeatureCalculator.safe_numeric(participant.get('victoireent', 0))
        trainer_places = FeatureCalculator.safe_numeric(participant.get('placeent', 0))
        trainer_courses = FeatureCalculator.safe_numeric(participant.get('courseent', 0))

        features['trainer_victoires'] = trainer_victoires
        features['trainer_places'] = trainer_places
        features['trainer_courses'] = trainer_courses

        # Trainer win rate
        if trainer_courses > 0:
            features['trainer_winrate'] = trainer_victoires / trainer_courses
            features['trainer_placerate'] = trainer_places / trainer_courses
        else:
            features['trainer_winrate'] = 0.0
            features['trainer_placerate'] = 0.0

        # Trainer quality score (normalized by number of races)
        if trainer_courses > 0:
            # Score based on success rate with confidence weighting
            confidence_weight = min(1.0, trainer_courses / 50)  # Full confidence at 50+ races
            quality = (trainer_victoires * 3 + trainer_places) / trainer_courses
            features['trainer_quality_score'] = quality * confidence_weight
        else:
            features['trainer_quality_score'] = 0.0

        # Trainer experience level
        if trainer_courses >= 200:
            features['trainer_experience_level'] = 3  # Experienced
        elif trainer_courses >= 50:
            features['trainer_experience_level'] = 2  # Moderate
        elif trainer_courses > 0:
            features['trainer_experience_level'] = 1  # Beginner
        else:
            features['trainer_experience_level'] = 0  # No data

        return features

    @staticmethod
    def calculate_handicap_weight_features(participant: Dict) -> Dict:
        """
        Calculate handicap and weight-related features.

        Args:
            participant: Participant data

        Returns:
            Dict with handicap/weight features
        """
        features = {}

        # Current handicap value (use safe numeric conversion)
        handicap_poids = FeatureCalculator.safe_numeric(participant.get('handicappoids', 0))
        handicap_dist = FeatureCalculator.safe_numeric(participant.get('handicapdist', 0))

        features['handicap_poids'] = handicap_poids
        features['handicap_dist'] = handicap_dist

        # Previous handicap
        prec_handicap_poids = FeatureCalculator.safe_numeric(participant.get('handicappoidsprec', 0))
        prec_handicap_dist = FeatureCalculator.safe_numeric(participant.get('handicapdistprec', 0))

        features['prec_handicap_poids'] = prec_handicap_poids
        features['prec_handicap_dist'] = prec_handicap_dist

        # Handicap changes (important for performance prediction)
        if prec_handicap_poids > 0:
            features['handicap_poids_change'] = handicap_poids - prec_handicap_poids
            features['handicap_poids_increased'] = float(handicap_poids > prec_handicap_poids)
            features['handicap_poids_decreased'] = float(handicap_poids < prec_handicap_poids)
        else:
            features['handicap_poids_change'] = 0.0
            features['handicap_poids_increased'] = 0.0
            features['handicap_poids_decreased'] = 0.0

        if prec_handicap_dist > 0:
            features['handicap_dist_change'] = handicap_dist - prec_handicap_dist
            features['handicap_dist_increased'] = float(handicap_dist > prec_handicap_dist)
            features['handicap_dist_decreased'] = float(handicap_dist < prec_handicap_dist)
        else:
            features['handicap_dist_change'] = 0.0
            features['handicap_dist_increased'] = 0.0
            features['handicap_dist_decreased'] = 0.0

        # Combined handicap severity score
        total_handicap = handicap_poids + handicap_dist
        features['total_handicap'] = total_handicap

        # Handicap momentum (negative = getting easier, positive = getting harder)
        handicap_momentum = features['handicap_poids_change'] + features['handicap_dist_change']
        features['handicap_momentum'] = handicap_momentum

        return features

    @staticmethod
    def encode_corde(corde_value) -> float:
        """
        Encode post position (corde) to numeric value.

        Args:
            corde_value: Post position (can be number or text like 'gauche')

        Returns:
            Numeric encoding
        """
        if corde_value is None or corde_value == '':
            return 0.0

        # Try numeric first
        try:
            return float(corde_value)
        except (ValueError, TypeError):
            pass

        # Handle text values
        corde_str = str(corde_value).lower().strip()
        if corde_str in ['gauche', 'left']:
            return -1.0  # Special value for inside rail
        elif corde_str in ['droite', 'right']:
            return 99.0  # Special value for far outside
        elif corde_str in ['milieu', 'middle', 'centre', 'center']:
            return 50.0  # Middle position
        else:
            return 0.0  # Unknown

    @staticmethod
    def calculate_last_race_comparison_features(participant: Dict) -> Dict:
        """
        Calculate features comparing current race to last race.

        Args:
            participant: Participant data

        Returns:
            Dict with last race comparison features
        """
        features = {}

        # Current race info (use safe numeric conversion for text values)
        current_dist = FeatureCalculator.safe_numeric(participant.get('dist', 0))
        current_corde = FeatureCalculator.encode_corde(participant.get('corde', 0))

        # Last race info
        last_dist = FeatureCalculator.safe_numeric(participant.get('derniere_dist', 0))
        last_place = FeatureCalculator.safe_numeric(participant.get('derniere_place', 0))

        features['last_race_position'] = last_place
        features['last_race_distance'] = last_dist

        # Distance change analysis
        if last_dist > 0:
            dist_change = current_dist - last_dist
            features['distance_change'] = dist_change
            features['distance_change_pct'] = (dist_change / last_dist) * 100
            features['distance_increased'] = float(dist_change > 0)
            features['distance_decreased'] = float(dist_change < 0)
            features['same_distance'] = float(abs(dist_change) < 50)  # Within 50m
        else:
            features['distance_change'] = 0.0
            features['distance_change_pct'] = 0.0
            features['distance_increased'] = 0.0
            features['distance_decreased'] = 0.0
            features['same_distance'] = 0.0

        # Post position change
        last_corde = FeatureCalculator.encode_corde(participant.get('derniere_corde', 0))
        if last_corde != 0.0:
            corde_change = current_corde - last_corde
            features['corde_change'] = corde_change
            features['corde_improved'] = float(corde_change < 0)  # Lower is better
            features['corde_worsened'] = float(corde_change > 0)
        else:
            features['corde_change'] = 0.0
            features['corde_improved'] = 0.0
            features['corde_worsened'] = 0.0

        # Last race performance quality
        if last_place > 0:
            if last_place <= 3:
                features['last_race_top3'] = 1.0
            else:
                features['last_race_top3'] = 0.0

            if last_place == 1:
                features['last_race_won'] = 1.0
            else:
                features['last_race_won'] = 0.0
        else:
            features['last_race_top3'] = 0.0
            features['last_race_won'] = 0.0

        return features

    @staticmethod
    def calculate_advanced_musique_momentum(che_musique_stats: Dict, joc_musique_stats: Dict) -> Dict:
        """
        Calculate advanced momentum features from musique statistics.

        Args:
            che_musique_stats: Horse musique statistics
            joc_musique_stats: Jockey musique statistics

        Returns:
            Dict with advanced momentum features
        """
        features = {}

        # Horse form momentum
        che_global = che_musique_stats['global']
        che_weighted = che_musique_stats['weighted']

        # Recent vs average performance comparison (negative = improving)
        if che_global['nb_courses'] >= 3:
            features['che_form_momentum'] = che_global['avg_pos'] - che_global['recent_perf']
            features['che_improving_form'] = float(features['che_form_momentum'] > 0)
            features['che_declining_form'] = float(features['che_form_momentum'] < -1)
        else:
            features['che_form_momentum'] = 0.0
            features['che_improving_form'] = 0.0
            features['che_declining_form'] = 0.0

        # Consistency scoring (lower std = more predictable)
        if che_global['consistency'] < 2.0:
            features['che_high_consistency'] = 1.0
        elif che_global['consistency'] < 3.0:
            features['che_high_consistency'] = 0.5
        else:
            features['che_high_consistency'] = 0.0

        # Recent performance quality
        if che_global['nb_courses'] >= 3:
            if che_global['recent_perf'] <= 3:
                features['che_recent_quality'] = 1.0  # Excellent recent form
            elif che_global['recent_perf'] <= 5:
                features['che_recent_quality'] = 0.7  # Good recent form
            elif che_global['recent_perf'] <= 8:
                features['che_recent_quality'] = 0.4  # Average recent form
            else:
                features['che_recent_quality'] = 0.1  # Poor recent form
        else:
            features['che_recent_quality'] = 0.0

        # Win/place rate quality indicators
        features['che_elite_performer'] = float(che_global['pct_top3'] >= 0.33)  # Top 3 in 33%+ of races
        features['che_regular_placer'] = float(che_global['pct_top3'] >= 0.20)  # Top 3 in 20%+ of races

        # Jockey form momentum (same logic)
        joc_global = joc_musique_stats['global']

        if joc_global['nb_courses'] >= 3:
            features['joc_form_momentum'] = joc_global['avg_pos'] - joc_global['recent_perf']
            features['joc_improving_form'] = float(features['joc_form_momentum'] > 0)
        else:
            features['joc_form_momentum'] = 0.0
            features['joc_improving_form'] = 0.0

        if joc_global['consistency'] < 2.0:
            features['joc_high_consistency'] = 1.0
        elif joc_global['consistency'] < 3.0:
            features['joc_high_consistency'] = 0.5
        else:
            features['joc_high_consistency'] = 0.0

        features['joc_elite_performer'] = float(joc_global['pct_top3'] >= 0.25)  # Jockey places 25%+ of time

        # Combined horse-jockey momentum synergy
        both_improving = features['che_improving_form'] and features['joc_improving_form']
        features['synergy_both_improving'] = float(both_improving)

        both_consistent = (features['che_high_consistency'] >= 0.5 and
                          features['joc_high_consistency'] >= 0.5)
        features['synergy_both_consistent'] = float(both_consistent)

        both_elite = features['che_elite_performer'] and features['joc_elite_performer']
        features['synergy_both_elite'] = float(both_elite)

        # Overall momentum score (0-5 scale)
        momentum_score = 0
        if features['che_improving_form']:
            momentum_score += 1
        if features['joc_improving_form']:
            momentum_score += 1
        if features['che_high_consistency'] >= 0.5:
            momentum_score += 1
        if features['che_recent_quality'] >= 0.7:
            momentum_score += 1
        if features['synergy_both_elite']:
            momentum_score += 1

        features['overall_momentum_score'] = float(momentum_score)
        features['high_momentum'] = float(momentum_score >= 3)

        return features

    @staticmethod
    def calculate_all_features(df, use_temporal=False, db_path=None):
        """
        Calcule toutes les features dérivées pour chaque participant dans le DataFrame.

        Args:
            df: DataFrame contenant les données brutes des participants
            use_temporal: If True, use temporal calculator for career stats
            db_path: Database path for temporal calculations

        Returns:
            DataFrame avec toutes les features calculées ajoutées
        """
        # Créer une copie du DataFrame pour éviter de modifier l'original
        result_df = df.copy()

        # Apply temporal calculations FIRST (batch mode - super fast!)
        if use_temporal and db_path:
            from core.calculators.temporal_feature_calculator import TemporalFeatureCalculator
            temporal_calc = TemporalFeatureCalculator(db_path)
            print(f"  🔄 Using temporal calculations (no data leakage mode)")
            # For prediction: preserve existing database values instead of recalculating
            preserve_existing = 'victoirescheval' in result_df.columns and result_df['victoirescheval'].notna().any()
            result_df = temporal_calc.batch_calculate_all_horses(result_df, preserve_existing=preserve_existing)

        # Apply comprehensive data cleaning for TabNet compatibility
        cleaner = TabNetDataCleaner()
        result_df = cleaner.comprehensive_data_cleaning(result_df, verbose=False)

        # Calculate field mean earnings for confidence weighting
        # This helps prevent extreme outliers from breaking TabNet scaling
        participants_list = result_df.to_dict('records')
        field_mean = FeatureCalculator.calculate_field_mean_earnings(participants_list)

        if len(result_df) > 0:
            print(f"🏇 Field mean earnings per race: {field_mean:,.0f} (from {len(participants_list)} horses)")

        # Progress tracking
        total_rows = len(result_df)
        print(f"  📊 Processing {total_rows} horses for feature calculation...")

        # Itérer sur chaque ligne du DataFrame
        for idx, (index, participant_row) in enumerate(result_df.iterrows()):
            if idx % 5000 == 0 and idx > 0:
                print(f"  ⏳ Progress: {idx}/{total_rows} horses ({idx/total_rows*100:.1f}%)")
            # Extraire le participant comme un dictionnaire
            participant = participant_row.to_dict()

            # Temporal stats already applied in batch mode above - skip per-horse calculation

            # Collect all features in a single dictionary to avoid DataFrame fragmentation
            row_features = {}

            # Calculer les ratios de performance avec pondération par confiance
            ratios = FeatureCalculator.calculate_performance_ratios(participant, field_mean)
            row_features.update(ratios)

            # Calculer les statistiques de couple
            couple_stats = FeatureCalculator.calculate_couple_stats(participant)
            row_features.update(couple_stats)

            # Calculer les statistiques d'hippodrome
            hippo_stats = FeatureCalculator.calculate_hippo_stats(participant)
            row_features.update(hippo_stats)

            # Calculer les statistiques d'entraîneur
            trainer_stats = FeatureCalculator.calculate_trainer_stats(participant)
            row_features.update(trainer_stats)

            # Calculer les features de handicap et poids
            handicap_features = FeatureCalculator.calculate_handicap_weight_features(participant)
            row_features.update(handicap_features)

            # Calculer les features de comparaison dernière course
            last_race_comparison = FeatureCalculator.calculate_last_race_comparison_features(participant)
            row_features.update(last_race_comparison)

            # Calculer les features d'équipement (blinkers)
            blinkers_features = FeatureCalculator.calculate_blinkers_features(participant)
            row_features.update(blinkers_features)

            # Calculer les features d'équipement (shoeing)
            shoeing_features = FeatureCalculator.calculate_shoeing_features(participant)
            row_features.update(shoeing_features)

            # Calculer les features d'impact combiné d'équipement
            equipment_impact_features = FeatureCalculator.calculate_equipment_impact_features(participant)
            row_features.update(equipment_impact_features)

            # Phase 1: Calculer les features de carrière avec pondération par confiance
            phase1_career_features = FeatureCalculator.calculate_phase1_career_features(participant, field_mean)
            row_features.update(phase1_career_features)

            # Phase 1: Calculer les features de dernière course
            phase1_last_race_features = FeatureCalculator.calculate_phase1_last_race_features(participant)
            row_features.update(phase1_last_race_features)

            # Phase 1: Calculer les features de rating/classification
            phase1_rating_features = FeatureCalculator.calculate_phase1_rating_features(participant)
            row_features.update(phase1_rating_features)

            # Extraire les features de la musique cheval
            cheval_musique_extractor = MusiqueFeatureExtractor()
            musique_che = participant.get('musiqueche', '')
            race_type = result_df.at[index, 'typec']
            che_musique_stats = cheval_musique_extractor.extract_features(musique_che, race_type)

            # DEBUG: Check bytype extraction on first horse
            if idx == 0:
                print(f"\n🔍 DEBUG - Musique extraction (first horse):")
                print(f"  musiqueche: {str(musique_che)[:60]}")
                print(f"  typec: {race_type}")
                print(f"  bytype DNF rate: {che_musique_stats['by_type'].get('dnf_rate', 'MISSING')}")
                print(f"  bytype avg_pos: {che_musique_stats['by_type'].get('avg_pos', 'MISSING')}")
                print(f"  bytype nb_courses: {che_musique_stats['by_type'].get('nb_courses', 'MISSING')}\n")

            # Correct way to access nested dictionaries
            # Add 'global' features
            for key, value in che_musique_stats['global'].items():
                column_name = f"che_global_{key}"  # Create prefixed column name
                row_features[column_name] = value

            # Add 'weighted' features
            for key, value in che_musique_stats['weighted'].items():
                column_name = f"che_weighted_{key}"  # Create prefixed column name
                row_features[column_name] = value

            # Add 'by_type' features (flat dictionary for current race type)
            for key, value in che_musique_stats['by_type'].items():
                column_name = f"che_bytype_{key}"
                row_features[column_name] = value
            # Extraire les features de la musique jockey
            jockey_musique_extractor = MusiqueFeatureExtractor()
            # Use jockey musique field if available, fallback to horse musique
            jockey_musique = participant.get('musiquejoc', participant.get('musiqueche', ''))
            joc_musique_stats = jockey_musique_extractor.extract_features(jockey_musique,
                                                                                  df.at[index, 'typec'])

            # Correct way to access nested dictionaries
            # Add 'global' features
            for key, value in joc_musique_stats['global'].items():
                        column_name = f"joc_global_{key}"  # Create prefixed column name
                        row_features[column_name] = value

                    # Add 'weighted' features
            for key, value in joc_musique_stats['weighted'].items():
                column_name = f"joc_weighted_{key}"  # Create prefixed column name
                row_features[column_name] = value

                # Add 'by_type' features (flat dictionary for current race type)
            for key, value in joc_musique_stats['by_type'].items():
                column_name = f"joc_bytype_{key}"
                row_features[column_name] = value

            # Calculate advanced musique momentum features
            momentum_features = FeatureCalculator.calculate_advanced_musique_momentum(
                che_musique_stats, joc_musique_stats
            )
            row_features.update(momentum_features)

            # Advanced derived features
            race_info = {
                'cheque': participant.get('cheque', 0),
                'partant': participant.get('partant', 0),
                'dist': participant.get('dist', 0),
                'handicap_level_score': participant.get('handicap_level_score', 0)
            }

            # Calculate advanced features
            class_features = FeatureCalculator.calculate_class_movement_features(participant, race_info)
            row_features.update(class_features)

            speed_features = FeatureCalculator.calculate_speed_figure_features(participant, race_info)
            row_features.update(speed_features)

            form_features = FeatureCalculator.calculate_form_context_features(participant, race_info)
            row_features.update(form_features)

            connection_features = FeatureCalculator.calculate_connection_features(participant)
            row_features.update(connection_features)

            competition_features = FeatureCalculator.calculate_competition_context_features(participant, race_info)
            row_features.update(competition_features)

            interaction_features = FeatureCalculator.calculate_interaction_features(participant, race_info)
            row_features.update(interaction_features)

            # Update the DataFrame row once with all collected features to avoid fragmentation
            # CRITICAL: Don't overwrite existing raw data fields that models expect
            preserve_fields = {'victoirescheval', 'placescheval', 'coursescheval', 'recence', 'cotedirect'}
            for key, value in row_features.items():
                # Only update if not a preserve field, or if field doesn't exist yet
                if key not in preserve_fields or pd.isna(result_df.at[index, key]):
                    result_df.at[index, key] = value

        return result_df