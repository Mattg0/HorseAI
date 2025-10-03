#!/usr/bin/env python3
"""
Phase 2 Advanced Feature Engineering Calculator

This module implements sophisticated derived features using newly synced data
to create powerful class analysis, speed figures, form context, and connection features.

Target: Improve R² from 0.0235 to 0.10-0.15 range with 15-20 high-impact features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import re
from decimal import Decimal


class Phase2FeatureCalculator:
    """
    Advanced feature calculator for Phase 2 derived features.
    Creates sophisticated features from race-level and participant-level data.
    """
    
    @staticmethod
    def safe_numeric(value: Any, default: float = 0.0) -> float:
        """Safely convert value to numeric, handling various types."""
        if pd.isna(value) or value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, str):
            if value.strip() in ('', 'N/A', 'NULL'):
                return default
            try:
                return float(value.replace(',', ''))
            except:
                return default
        return default
    
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
        """
        Calculate class movement analysis features.
        
        Features created:
        - class_drop_pct: Percentage drop in race purse
        - purse_per_starter: Current race purse per starter
        - purse_ratio: Current vs previous purse ratio
        - moving_up_in_class: Boolean indicator for class rise
        - class_shock_indicator: Large class change indicator
        """
        features = {}
        
        # Current race purse
        current_purse = Phase2FeatureCalculator.safe_numeric(race_info.get('cheque', 0))
        
        # Previous race purse
        prev_purse = Phase2FeatureCalculator.safe_numeric(participant.get('dernierealloc', 0))
        
        # Field size
        field_size = Phase2FeatureCalculator.safe_numeric(race_info.get('partant', 1))
        
        # 1. Class drop percentage
        if prev_purse > 0:
            features['class_drop_pct'] = max(0, (prev_purse - current_purse) / prev_purse)
        else:
            features['class_drop_pct'] = 0.0
            
        # 2. Purse per starter (race competitiveness indicator)
        if field_size > 0 and current_purse > 0 and np.isfinite(current_purse):
            features['purse_per_starter'] = current_purse / field_size
        else:
            features['purse_per_starter'] = 0.0
        
        # 3. Purse ratio (class level comparison)
        if prev_purse > 0:
            features['purse_ratio'] = current_purse / prev_purse
        else:
            features['purse_ratio'] = 1.0
            
        # 4. Moving up in class (significant class rise)
        features['moving_up_in_class'] = 1.0 if features['purse_ratio'] > 1.5 else 0.0
        
        # 5. Class shock indicator (large class change in either direction)
        ratio = features['purse_ratio']
        features['class_shock_indicator'] = 1.0 if (ratio > 2.0 or ratio < 0.5) else 0.0
        
        # 6. Handicap level change (if available)
        current_handicap_score = Phase2FeatureCalculator.safe_numeric(race_info.get('handicap_level_score', 0))
        features['handicap_level_change'] = current_handicap_score  # Previous not available yet
        
        return features
    
    @staticmethod
    def calculate_speed_figure_features(participant: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate speed figure proxy features.
        
        Features created:
        - time_per_distance: Time efficiency metric
        - time_vs_record: Current time vs personal best
        - pace_rating: Standardized pace rating
        - speed_figure_proxy: Combined speed metric
        """
        features = {}
        
        # Current race time and distance
        current_time_str = Phase2FeatureCalculator.safe_string(participant.get('tempstot', ''))
        current_distance = Phase2FeatureCalculator.safe_numeric(race_info.get('dist', 0))
        
        # Personal record time
        record_time_str = Phase2FeatureCalculator.safe_string(participant.get('recordG', ''))
        
        # Parse times
        current_time = Phase2FeatureCalculator.parse_time_to_seconds(current_time_str)
        record_time = Phase2FeatureCalculator.parse_time_to_seconds(record_time_str)
        
        # 1. Time per distance (speed efficiency)
        if current_time and current_distance > 0:
            features['time_per_distance'] = current_time / current_distance
        else:
            features['time_per_distance'] = 0.0
            
        # 2. Time vs record ratio
        if current_time and record_time and record_time > 0:
            features['time_vs_record'] = current_time / record_time
        else:
            features['time_vs_record'] = 1.0
            
        # 3. Pace rating (normalized by distance)
        if current_time and current_distance > 0:
            # Normalize to 2000m standard
            standard_distance = 2000
            normalized_time = (current_time / current_distance) * standard_distance
            # Convert to rating (lower time = higher rating)
            features['pace_rating'] = max(0, 200 - normalized_time / 60 * 100)
        else:
            features['pace_rating'] = 100.0  # Neutral rating
            
        # 4. Speed figure proxy (combined metric)
        time_component = 1.0 / max(features['time_vs_record'], 0.1)  # Better time = higher score
        distance_component = features['time_per_distance'] if features['time_per_distance'] > 0 else 1.0
        features['speed_figure_proxy'] = time_component / max(distance_component, 0.001) * 100
        
        return features
    
    @staticmethod
    def calculate_form_context_features(participant: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate form context analysis features.
        
        Features created:
        - market_confidence_shift: Change in market confidence
        - form_momentum: Recent form trend indicator  
        - bounce_risk: Risk of poor performance after good race
        - layoff_distance_interaction: Recency and distance change interaction
        """
        features = {}
        
        # Current and previous odds
        current_odds = Phase2FeatureCalculator.safe_numeric(participant.get('coteprob', 0))
        prev_odds = Phase2FeatureCalculator.safe_numeric(participant.get('dernierecote', 0))
        
        # Previous performance
        prev_position_str = Phase2FeatureCalculator.safe_string(participant.get('derniereplace', ''))
        
        # Distance comparison
        current_distance = Phase2FeatureCalculator.safe_numeric(race_info.get('dist', 0))
        prev_distance = Phase2FeatureCalculator.safe_numeric(participant.get('dernieredist', 0))
        
        # Recency
        recency = Phase2FeatureCalculator.safe_numeric(participant.get('recence', 0))
        
        # 1. Market confidence shift
        if prev_odds > 0 and current_odds > 0:
            # Lower odds = higher confidence, so inverse relationship
            features['market_confidence_shift'] = prev_odds / current_odds
        else:
            features['market_confidence_shift'] = 1.0
            
        # 2. Last race improvement indicator
        try:
            prev_position = int(prev_position_str) if prev_position_str.isdigit() else 10
            # Good previous performance = low number
            features['last_race_improvement'] = max(0, 15 - prev_position) / 15
        except:
            features['last_race_improvement'] = 0.5
            
        # 3. Bounce risk (after good performance)
        prev_pos_num = 10
        try:
            if prev_position_str.isdigit():
                prev_pos_num = int(prev_position_str)
        except:
            pass
            
        # Risk increases after very good performance (positions 1-3)
        features['bounce_risk'] = 1.0 if prev_pos_num <= 3 else 0.0
        
        # 4. Layoff vs distance interaction
        if prev_distance > 0:
            distance_change_pct = abs(current_distance - prev_distance) / prev_distance
            # Higher risk with long layoff and distance change
            features['layoff_distance_interaction'] = (recency / 30) * distance_change_pct
        else:
            features['layoff_distance_interaction'] = recency / 60  # Just recency factor
            
        return features
    
    @staticmethod
    def calculate_connection_features(participant: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate trainer/connection change features.
        
        Features created:
        - trainer_change: Boolean indicator of trainer change
        - connection_stability: Stability in horse connections
        """
        features = {}
        
        # Current and previous trainer
        current_trainer = Phase2FeatureCalculator.safe_string(participant.get('entraineur', ''))
        prev_trainer = Phase2FeatureCalculator.safe_string(participant.get('dernierEnt', ''))
        
        # 1. Trainer change indicator
        if current_trainer and prev_trainer and current_trainer != prev_trainer:
            features['trainer_change'] = 1.0
        else:
            features['trainer_change'] = 0.0
            
        # 2. Connection stability (inverse of changes)
        features['connection_stability'] = 1.0 - features['trainer_change']
        
        return features
    
    @staticmethod
    def calculate_competition_context_features(participant: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate competition context features.
        
        Features created:
        - field_size_change: Change in competition field size
        - distance_comfort: Distance suitability indicator
        - competition_level_shift: Change in competition level
        """
        features = {}
        
        # Field sizes
        current_field = Phase2FeatureCalculator.safe_numeric(race_info.get('partant', 0))
        prev_field = Phase2FeatureCalculator.safe_numeric(participant.get('derniernbpartants', 0))
        
        # Distances  
        current_distance = Phase2FeatureCalculator.safe_numeric(race_info.get('dist', 0))
        prev_distance = Phase2FeatureCalculator.safe_numeric(participant.get('dernieredist', 0))
        
        # 1. Field size change
        if prev_field > 0:
            features['field_size_change'] = (current_field - prev_field) / prev_field
        else:
            features['field_size_change'] = 0.0
            
        # 2. Distance comfort (negative penalty for big changes)
        if prev_distance > 0:
            distance_change_pct = abs(current_distance - prev_distance) / prev_distance
            features['distance_comfort'] = max(0, 1.0 - distance_change_pct)
        else:
            features['distance_comfort'] = 0.5  # Neutral when no previous distance
            
        # 3. Competition level shift (field size impact on competitiveness)  
        features['competition_level_shift'] = current_field / 20.0  # Normalize around typical field size
        
        return features
    
    @staticmethod
    def calculate_interaction_features(participant: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate feature interaction combinations based on top-performing features.
        
        Features created:
        - recence_x_class_drop: Layoff with class change interaction
        - trainer_change_x_recence: New trainer with fresh horse
        - distance_change_x_recence: Distance change with layoff
        """
        features = {}
        
        # Get base features
        recency = Phase2FeatureCalculator.safe_numeric(participant.get('recence', 0))
        
        # Calculate class drop for interaction
        current_purse = Phase2FeatureCalculator.safe_numeric(race_info.get('cheque', 0))
        prev_purse = Phase2FeatureCalculator.safe_numeric(participant.get('dernierealloc', 0))
        
        class_drop = 0.0
        if prev_purse > 0:
            class_drop = max(0, (prev_purse - current_purse) / prev_purse)
            
        # Trainer change
        current_trainer = Phase2FeatureCalculator.safe_string(participant.get('entraineur', ''))
        prev_trainer = Phase2FeatureCalculator.safe_string(participant.get('dernierEnt', ''))
        trainer_change = 1.0 if (current_trainer and prev_trainer and current_trainer != prev_trainer) else 0.0
        
        # Distance change
        current_distance = Phase2FeatureCalculator.safe_numeric(race_info.get('dist', 0))
        prev_distance = Phase2FeatureCalculator.safe_numeric(participant.get('dernieredist', 0))
        
        distance_change = 0.0
        if prev_distance > 0:
            distance_change = abs(current_distance - prev_distance) / prev_distance
        
        # 1. Recency × Class Drop (powerful combination - fresh horse dropping in class)
        features['recence_x_class_drop'] = (recency / 30) * class_drop
        
        # 2. Trainer Change × Recency (new connections with fresh horse)
        features['trainer_change_x_recence'] = trainer_change * (recency / 30)
        
        # 3. Distance Change × Recency (distance experiments after layoff)
        features['distance_change_x_recence'] = distance_change * (recency / 30)
        
        return features
    
    @staticmethod
    def calculate_all_phase2_features(participant: Dict[str, Any], race_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate all Phase 2 advanced features for a single participant.
        
        Args:
            participant: Dictionary with participant data
            race_info: Dictionary with race-level information
            
        Returns:
            Dictionary with all Phase 2 derived features
        """
        all_features = {}
        
        # Calculate each feature category
        class_features = Phase2FeatureCalculator.calculate_class_movement_features(participant, race_info)
        speed_features = Phase2FeatureCalculator.calculate_speed_figure_features(participant, race_info)
        form_features = Phase2FeatureCalculator.calculate_form_context_features(participant, race_info)
        connection_features = Phase2FeatureCalculator.calculate_connection_features(participant)
        competition_features = Phase2FeatureCalculator.calculate_competition_context_features(participant, race_info)
        interaction_features = Phase2FeatureCalculator.calculate_interaction_features(participant, race_info)
        
        # Combine all features
        all_features.update(class_features)
        all_features.update(speed_features)
        all_features.update(form_features)
        all_features.update(connection_features)
        all_features.update(competition_features)
        all_features.update(interaction_features)
        
        return all_features
    
    @staticmethod
    def get_phase2_feature_names() -> list:
        """Get list of all Phase 2 feature names."""
        return [
            # Class movement (6 features)
            'class_drop_pct', 'purse_per_starter', 'purse_ratio', 'moving_up_in_class',
            'class_shock_indicator', 'handicap_level_change',
            
            # Speed figures (4 features)  
            'time_per_distance', 'time_vs_record', 'pace_rating', 'speed_figure_proxy',
            
            # Form context (4 features)
            'market_confidence_shift', 'last_race_improvement', 'bounce_risk', 'layoff_distance_interaction',
            
            # Connections (2 features)
            'trainer_change', 'connection_stability',
            
            # Competition context (3 features)
            'field_size_change', 'distance_comfort', 'competition_level_shift',
            
            # Interactions (3 features)
            'recence_x_class_drop', 'trainer_change_x_recence', 'distance_change_x_recence'
        ]


def test_phase2_features():
    """Test Phase 2 feature calculation with sample data."""
    
    print("=== TESTING PHASE 2 FEATURE ENGINEERING ===")
    
    # Sample data
    sample_participant = {
        'derniereplace': '3',
        'dernierecote': '4.50',
        'tempstot': "2'42\"53",
        'dernierealloc': '25000',
        'dernieredist': '2100',
        'derniernbpartants': '16',
        'recordG': "2'38\"12",
        'entraineur': 'TRAINER_A',
        'dernierEnt': 'TRAINER_B',
        'recence': '14',
        'coteprob': '3.80'
    }
    
    sample_race = {
        'cheque': '35000',
        'partant': '18',
        'dist': '2000',
        'handicap_level_score': '650'
    }
    
    # Calculate features
    features = Phase2FeatureCalculator.calculate_all_phase2_features(sample_participant, sample_race)
    
    print(f"\nCalculated {len(features)} Phase 2 features:")
    print("="*50)
    
    # Group by category
    categories = {
        'Class Movement': ['class_drop_pct', 'purse_per_starter', 'purse_ratio', 'moving_up_in_class'],
        'Speed Figures': ['time_per_distance', 'time_vs_record', 'pace_rating', 'speed_figure_proxy'],
        'Form Context': ['market_confidence_shift', 'last_race_improvement', 'bounce_risk'],
        'Connections': ['trainer_change', 'connection_stability'],
        'Competition': ['field_size_change', 'distance_comfort', 'competition_level_shift'],
        'Interactions': ['recence_x_class_drop', 'trainer_change_x_recence', 'distance_change_x_recence']
    }
    
    for category, feature_list in categories.items():
        print(f"\n{category}:")
        for feature_name in feature_list:
            if feature_name in features:
                value = features[feature_name]
                print(f"  {feature_name:25s}: {value:.4f}")
    
    print(f"\nTotal features: {len(features)}")
    print("✅ Phase 2 feature engineering test completed!")
    
    return features


if __name__ == "__main__":
    test_phase2_features()