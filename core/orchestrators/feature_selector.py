#!/usr/bin/env python3
"""
Feature selector for dual pipeline system.
Provides model-specific feature selection while maintaining consistency.
"""

from typing import List, Dict, Set, Tuple
import pandas as pd
import logging

class ModelFeatureSelector:
    """
    Manages feature selection for different model types while ensuring
    consistency between training and prediction phases.
    """
    
    def __init__(self, config=None):
        """
        Initialize feature selector with configuration.
        
        Args:
            config: AppConfig instance for feature configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define model-specific feature strategies
        self._define_feature_strategies()
    
    def _define_feature_strategies(self):
        """Define feature selection strategies for different model types."""
        
        # LSTM features (embeddings + sequential/static architecture)
        self.lstm_features = {
            'sequential': [
                # Features that vary race-to-race in the 5-race sequence
                'final_position', 'cotedirect', 'dist', 'recence',
                'derniereplace', 'dernierecote', 'dernieredist', 'derniernbpartants',
                'gainsAnneeEnCours',  # Grows over time
                # Embeddings that can vary by sequence position
                'horse_emb_0', 'horse_emb_1', 'horse_emb_2', 'horse_emb_3',
                'horse_emb_4', 'horse_emb_5', 'horse_emb_6', 'horse_emb_7',
                'jockey_emb_0', 'jockey_emb_1', 'jockey_emb_2', 'jockey_emb_3',
                'jockey_emb_4', 'jockey_emb_5', 'jockey_emb_6', 'jockey_emb_7',
                # Sequential performance indicators (trends over time)
                'che_global_avg_pos', 'che_global_recent_perf', 'che_global_consistency',
                'joc_global_avg_pos', 'joc_global_recent_perf'
            ],
            'static': [
                # Features that remain constant across the sequence
                'age', 'partant', 'temperature', 'natpis', 'typec', 'meteo', 'corde',
                # Phase 1: Career totals (constant per horse)
                'victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere',
                'vha',  # Official rating (changes slowly)
                # Course and couple embeddings (static context)
                'couple_emb_0', 'couple_emb_1', 'couple_emb_2', 'couple_emb_3',
                'couple_emb_4', 'couple_emb_5', 'couple_emb_6', 'couple_emb_7',
                'course_emb_0', 'course_emb_1', 'course_emb_2', 'course_emb_3',
                'course_emb_4', 'course_emb_5', 'course_emb_6', 'course_emb_7',
                # Equipment features (remain constant within sequence)
                'blinkers_first_time', 'has_blinkers', 'has_standard_blinkers',
                'has_australian_blinkers', 'blinkers_added', 'blinkers_removed',
                'blinkers_type_changed', 'blinkers_any_change', 'blinkers_high_impact_change',
                'shoeing_first_time', 'fully_shod', 'front_shod_only', 'back_shod_only',
                'barefoot', 'shoeing_changed', 'barefoot_to_shod', 'shod_to_barefoot',
                'major_shoeing_change', 'shoeing_complexity', 'multiple_equipment_changes',
                'equipment_momentum_score', 'high_equipment_momentum', 'multiple_first_time_changes',
                'equipment_optimization_score'
            ]
        }
        
        # RF/TabNet features (raw domain features only - NO EMBEDDINGS)
        self.domain_features = [
            # Raw musique-derived performance features (horse)
            'che_global_avg_pos', 'che_global_recent_perf', 'che_global_trend',
            'che_global_consistency', 'che_global_pct_top3', 'che_global_nb_courses',
            'che_weighted_avg_pos', 'che_weighted_recent_perf', 'che_weighted_consistency',
            'che_weighted_trend', 'che_weighted_pct_top3',
            'che_bytype_avg_pos', 'che_bytype_recent_perf', 'che_bytype_trend',
            'che_bytype_consistency', 'che_bytype_pct_top3', 'che_bytype_nb_courses',
            
            # Jockey raw performance features
            'joc_global_avg_pos', 'joc_global_recent_perf', 'joc_global_trend',
            'joc_global_consistency', 'joc_global_pct_top3', 'joc_global_nb_courses',
            'joc_weighted_avg_pos', 'joc_weighted_recent_perf', 'joc_weighted_consistency',
            'joc_weighted_trend', 'joc_weighted_pct_top3',
            'joc_bytype_avg_pos', 'joc_bytype_recent_perf', 'joc_bytype_trend',
            'joc_bytype_consistency', 'joc_bytype_pct_top3', 'joc_bytype_nb_courses',
            
            # Raw couple statistics (not embeddings)
            'nbCourseCouple', 'nbVictCouple', 'nbPlaceCouple', 'TxVictCouple',
            'nbCourseCoupleHippo', 'nbVictCoupleHippo', 'nbPlaceCoupleHippo',
            'TxVictCoupleHippo', 'TxPlaceCoupleHippo',
            
            # Career performance (available in database)
            'victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere',
            'ratio_victoires', 'ratio_places', 'gains_par_course', 'career_strike_rate',
            'efficacite_couple', 'regularite_couple', 'progression_couple',
            
            # Track-specific performance (available in database)
            'perf_cheval_hippo', 'perf_jockey_hippo',
            'pourcVictChevalHippo', 'pourcPlaceChevalHippo',
            'pourcVictJockHippo', 'pourcPlaceJockHippo',
            
            # Current year earnings (available in database)
            'gainsAnneeEnCours', 'gainsAnneePrecedente', 'cheque', 'dernierealloc',
            'earnings_per_race',
            
            # Last race performance (single values)
            'derniereplace', 'dernierecote', 'dernieredist', 'derniernbpartants',
            
            # Official ratings and classifications
            'vha', 'txreclam', 'dernierTxreclam',
            
            # Equipment change features (high-impact domain features)
            'blinkers_first_time', 'has_blinkers', 'has_standard_blinkers',
            'has_australian_blinkers', 'blinkers_added', 'blinkers_removed',
            'blinkers_type_changed', 'blinkers_any_change', 'blinkers_high_impact_change',
            'shoeing_first_time', 'fully_shod', 'front_shod_only', 'back_shod_only',
            'barefoot', 'shoeing_changed', 'barefoot_to_shod', 'shod_to_barefoot',
            'major_shoeing_change', 'shoeing_complexity', 'multiple_equipment_changes',
            'equipment_momentum_score', 'high_equipment_momentum', 'multiple_first_time_changes',
            'equipment_optimization_score',
            
            # Core racing features
            'recence', 'age', 'cotedirect', 'coteprob',
            'handicapDistance', 'handicapPoids', 'poidmont',
            
            # Race context features (available in database)
            'temperature', 'forceVent', 'natpis', 'typec', 'meteo', 'corde',
            'dist', 'partant',
            
            # Phase 2: Advanced derived features (22 features)
            # Class movement analysis (6 features)
            'class_drop_pct', 'purse_per_starter', 'purse_ratio', 'moving_up_in_class',
            'class_shock_indicator', 'handicap_level_change',
            # Speed figure proxies (4 features)  
            'time_per_distance', 'time_vs_record', 'pace_rating', 'speed_figure_proxy',
            # Form context analysis (4 features)
            'market_confidence_shift', 'last_race_improvement', 'bounce_risk', 'layoff_distance_interaction',
            # Connection changes (2 features)
            'trainer_change', 'connection_stability',
            # Competition context (3 features)
            'field_size_change', 'distance_comfort', 'competition_level_shift',
            # Feature interactions (3 features)
            'recence_x_class_drop', 'trainer_change_x_recence', 'distance_change_x_recence'
        ]
        
        # Tabnet-specific features (subset of domain features optimized for TabNet)
        self.tabnet_features = [
            # Core performance indicators (expanded)
            'che_global_avg_pos', 'che_global_recent_perf', 'che_global_consistency',
            'che_global_pct_top3', 'che_weighted_avg_pos', 'che_weighted_recent_perf',
            'joc_global_avg_pos', 'joc_global_recent_perf', 'joc_global_consistency',
            'joc_global_pct_top3', 'joc_weighted_avg_pos', 'joc_weighted_recent_perf',
            
            # Career performance (key features)
            'victoirescheval', 'placescheval', 'coursescheval',
            'ratio_victoires', 'ratio_places', 'gains_par_course',
            
            # Track-specific performance
            'perf_cheval_hippo', 'perf_jockey_hippo',
            'pourcVictChevalHippo', 'pourcPlaceChevalHippo',
            
            # Couple performance
            'efficacite_couple', 'TxVictCouple', 'nbVictCouple',
            
            # Current year performance
            'gainsAnneeEnCours',
            
            # High-impact equipment features
            'blinkers_first_time', 'blinkers_high_impact_change', 'equipment_momentum_score',
            'major_shoeing_change', 'barefoot_to_shod', 'equipment_optimization_score',
            
            # Core racing features
            'recence', 'age', 'cotedirect', 'handicapDistance', 'temperature',
            'natpis', 'typec', 'dist', 'coteprob',
            
            # Phase 2: High-impact derived features for TabNet
            'class_drop_pct', 'purse_ratio', 'moving_up_in_class',
            'speed_figure_proxy', 'market_confidence_shift', 'trainer_change',
            'distance_comfort', 'recence_x_class_drop'
        ]

    def get_model_features(self, model_type: str, df: pd.DataFrame, 
                          feature_mode: str = None) -> List[str]:
        """
        Get appropriate feature list for the specified model type.
        
        Args:
            model_type: Type of model ('rf', 'tabnet')
            df: DataFrame to check for available features
            feature_mode: Optional override for feature selection mode
        
        Returns:
            List of feature names appropriate for the model type
        """
        model_type = model_type.lower()
        available_columns = set(df.columns)
        
        if model_type == 'lstm':
            # LSTM uses both sequential and static features (with embeddings)
            sequential = [f for f in self.lstm_features['sequential'] if f in available_columns]
            static = [f for f in self.lstm_features['static'] if f in available_columns]
            
            self.logger.info(f"LSTM features: {len(sequential)} sequential, {len(static)} static")
            return {'sequential': sequential, 'static': static}
            
        elif model_type in ['rf', 'random_forest']:
            # Random Forest uses raw domain features (no embeddings)
            features = [f for f in self.domain_features if f in available_columns]
            self.logger.info(f"RF features: {len(features)} domain features (no embeddings)")
            return features
            
        elif model_type == 'tabnet':
            # TabNet uses optimized subset of domain features
            features = [f for f in self.tabnet_features if f in available_columns]
            self.logger.info(f"TabNet features: {len(features)} optimized domain features")
            return features
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Only 'rf' and 'tabnet' are supported.")
    
    def get_features_to_drop(self, model_type: str, df: pd.DataFrame) -> List[str]:
        """
        Get list of features that should be dropped for the specified model type.
        
        Args:
            model_type: Type of model
            df: DataFrame with all available features
        
        Returns:
            List of feature names to drop
        """
        model_type = model_type.lower()
        all_columns = set(df.columns)
        
        # Get features that the model should keep
        keep_features = self.get_model_features(model_type, df)
        
        if isinstance(keep_features, dict):
            # LSTM case - flatten sequential and static features
            keep_features = set(keep_features['sequential'] + keep_features['static'])
        else:
            keep_features = set(keep_features)
        
        # Always keep essential metadata columns (numero removed - should not be a feature)
        essential_columns = {
            'comp', 'idche', 'jour', 'final_position', 'cl'
        }
        keep_features.update(essential_columns)
        
        # Features to drop are all columns except the ones we want to keep
        features_to_drop = all_columns - keep_features
        
        return list(features_to_drop)
    
    def apply_model_specific_filtering(self, df: pd.DataFrame, model_type: str,
                                     keep_metadata: bool = True) -> pd.DataFrame:
        """
        Apply model-specific feature filtering to a DataFrame.
        
        Args:
            df: DataFrame with all features
            model_type: Type of model to filter features for
            keep_metadata: Whether to keep essential metadata columns
        
        Returns:
            Filtered DataFrame with model-appropriate features
        """
        filtered_df = df.copy()
        
        # Get features to drop for this model type
        features_to_drop = self.get_features_to_drop(model_type, df)
        
        # Remove features that don't exist in the DataFrame
        features_to_drop = [f for f in features_to_drop if f in filtered_df.columns]
        
        if features_to_drop:
            filtered_df = filtered_df.drop(columns=features_to_drop)
            self.logger.info(f"Dropped {len(features_to_drop)} features for {model_type} model")
        
        return filtered_df
    
    def validate_feature_consistency(self, train_features: List[str], 
                                   predict_features: List[str]) -> Tuple[bool, Dict]:
        """
        Validate that training and prediction features are consistent.
        
        Args:
            train_features: Features used during training
            predict_features: Features available during prediction
        
        Returns:
            Tuple of (is_consistent, validation_report)
        """
        train_set = set(train_features)
        predict_set = set(predict_features)
        
        missing_features = train_set - predict_set
        extra_features = predict_set - train_set
        
        is_consistent = len(missing_features) == 0
        
        validation_report = {
            'is_consistent': is_consistent,
            'missing_features': list(missing_features),
            'extra_features': list(extra_features),
            'common_features': list(train_set & predict_set),
            'train_feature_count': len(train_features),
            'predict_feature_count': len(predict_features)
        }
        
        return is_consistent, validation_report
    
    def get_feature_importance_mapping(self, model_type: str) -> Dict[str, str]:
        """
        Get mapping of features to their domain meaning for interpretability.
        
        Args:
            model_type: Type of model
        
        Returns:
            Dictionary mapping feature names to human-readable descriptions
        """
        return {
            # Equipment features
            'blinkers_first_time': 'First-time blinkers (high-impact positive)',
            'blinkers_high_impact_change': 'High-impact blinkers change',
            'equipment_momentum_score': 'Combined equipment change momentum',
            'major_shoeing_change': 'Major shoeing pattern change',
            'barefoot_to_shod': 'Barefoot to shod transition',
            
            # Performance features
            'che_global_avg_pos': 'Horse average finishing position',
            'che_global_recent_perf': 'Horse recent performance trend',
            'joc_global_avg_pos': 'Jockey average finishing position',
            'ratio_victoires': 'Horse win percentage',
            'efficacite_couple': 'Horse-jockey combination effectiveness',
            
            # Core features
            'recence': 'Days since last race',
            'age': 'Horse age',
            'cotedirect': 'Direct odds',
            'temperature': 'Race day temperature',
            'dist': 'Race distance'
        }