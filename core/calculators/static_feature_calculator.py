from typing import Dict, List, Union, Optional
import pandas as pd
import numpy as np
import json
from core.calculators.musique_calculation import MusiqueFeatureExtractor
from core.calculators.phase2_feature_calculator import Phase2FeatureCalculator
from core.data_cleaning.tabnet_cleaner import TabNetDataCleaner

class FeatureCalculator:
    """
    Module de calcul des features dérivées pour les courses hippiques.
    Peut être utilisé à la fois pour l'entraînement et la prédiction.
    """
    
    @staticmethod
    def calculate_confidence_weighted_earnings(gains_carriere: float, courses_cheval: float, field_mean: float = 100000) -> float:
        """
        Calculate earnings per race with confidence weighting based on sample size.

        Horses with few races get shrunk toward field mean to prevent extreme outliers.
        Horses with many races (20+) use actual earnings per race.

        This fixes TabNet scaling issues caused by horses with minimal experience
        but high earnings creating extreme gains_par_course values.

        Args:
            gains_carriere: Total career earnings
            courses_cheval: Number of career races
            field_mean: Typical earnings per race (default 100k)

        Returns:
            Confidence-weighted earnings per race

        Examples:
            - Horse with 1 race, 675k earnings:
              Raw = 675,000, Confidence = 0.05, Result = ~67,500
            - Horse with 20 races, 150k avg:
              Raw = 150,000, Confidence = 1.0, Result = 150,000
        """
        if courses_cheval == 0:
            return 0.0

        # Ensure non-negative inputs
        gains_carriere = max(0.0, gains_carriere)
        field_mean = max(0.0, field_mean)

        # Calculate raw earnings per race
        raw_earnings_per_race = gains_carriere / courses_cheval

        # Confidence factor: 0 to 1 based on number of races
        # Plateaus at 20 races (100% confidence)
        confidence = min(courses_cheval / 20.0, 1.0)

        # Weight between individual estimate and field mean
        # Low confidence = closer to field mean
        # High confidence = use actual performance
        weighted_value = (raw_earnings_per_race * confidence) + (field_mean * (1 - confidence))

        # Final safety check
        return max(0.0, weighted_value)

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
        Validate earnings features to catch extreme outliers that could break models.

        Args:
            features: Dictionary of calculated features
            feature_name: Name of the feature to validate

        Raises:
            AssertionError: If feature values are outside reasonable bounds
        """
        if feature_name in features:
            value = features[feature_name]

            # Validate range
            assert value >= 0, f"Negative {feature_name}: {value}"

            # With confidence weighting, values should be much more reasonable
            # But allow some flexibility for genuinely high-earning experienced horses
            if value > 500000:
                print(f"🚨 Very high {feature_name}: {value:,.0f} (confidence weighting may need adjustment)")
            elif value > 300000:
                print(f"⚠️  High {feature_name}: {value:,.0f} (monitoring)")

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
                'gains_par_course': FeatureCalculator.calculate_confidence_weighted_earnings(
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

        # Earnings per race - now using confidence weighting to prevent TabNet scaling issues
        features['earnings_per_race'] = FeatureCalculator.calculate_confidence_weighted_earnings(
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
    def calculate_all_features(df):
        """
        Calcule toutes les features dérivées pour chaque participant dans le DataFrame.

        Args:
            df: DataFrame contenant les données brutes des participants

        Returns:
            DataFrame avec toutes les features calculées ajoutées
        """
        # Créer une copie du DataFrame pour éviter de modifier l'original
        result_df = df.copy()

        # Apply comprehensive data cleaning for TabNet compatibility
        cleaner = TabNetDataCleaner()
        result_df = cleaner.comprehensive_data_cleaning(result_df, verbose=False)

        # Calculate field mean earnings for confidence weighting
        # This helps prevent extreme outliers from breaking TabNet scaling
        participants_list = result_df.to_dict('records')
        field_mean = FeatureCalculator.calculate_field_mean_earnings(participants_list)

        if len(result_df) > 0:
            print(f"🏇 Field mean earnings per race: {field_mean:,.0f} (from {len(participants_list)} horses)")

        # Itérer sur chaque ligne du DataFrame
        for index, participant_row in result_df.iterrows():
            # Extraire le participant comme un dictionnaire
            participant = participant_row.to_dict()

            # Calculer les ratios de performance avec pondération par confiance
            ratios = FeatureCalculator.calculate_performance_ratios(participant, field_mean)
            for key, value in ratios.items():
                result_df.at[index, key] = value

            # Calculer les statistiques de couple
            couple_stats = FeatureCalculator.calculate_couple_stats(participant)
            for key, value in couple_stats.items():
                result_df.at[index, key] = value

            # Calculer les statistiques d'hippodrome
            hippo_stats = FeatureCalculator.calculate_hippo_stats(participant)
            for key, value in hippo_stats.items():
                result_df.at[index, key] = value

            # Calculer les features d'équipement (blinkers)
            blinkers_features = FeatureCalculator.calculate_blinkers_features(participant)
            for key, value in blinkers_features.items():
                result_df.at[index, key] = value

            # Calculer les features d'équipement (shoeing)
            shoeing_features = FeatureCalculator.calculate_shoeing_features(participant)
            for key, value in shoeing_features.items():
                result_df.at[index, key] = value

            # Calculer les features d'impact combiné d'équipement
            equipment_impact_features = FeatureCalculator.calculate_equipment_impact_features(participant)
            for key, value in equipment_impact_features.items():
                result_df.at[index, key] = value

            # Phase 1: Calculer les features de carrière avec pondération par confiance
            phase1_career_features = FeatureCalculator.calculate_phase1_career_features(participant, field_mean)
            for key, value in phase1_career_features.items():
                result_df.at[index, key] = value

            # Phase 1: Calculer les features de dernière course
            phase1_last_race_features = FeatureCalculator.calculate_phase1_last_race_features(participant)
            for key, value in phase1_last_race_features.items():
                result_df.at[index, key] = value

            # Phase 1: Calculer les features de rating/classification
            phase1_rating_features = FeatureCalculator.calculate_phase1_rating_features(participant)
            for key, value in phase1_rating_features.items():
                result_df.at[index, key] = value

            # Extraire les features de la musique cheval
            cheval_musique_extractor = MusiqueFeatureExtractor()
            che_musique_stats = cheval_musique_extractor.extract_features(participant['musiqueche'], df.at[index, 'typec'])

            # Correct way to access nested dictionaries
            # Add 'global' features
            for key, value in che_musique_stats['global'].items():
                column_name = f"che_global_{key}"  # Create prefixed column name
                result_df.at[index, column_name] = value

            # Add 'weighted' features
            for key, value in che_musique_stats['weighted'].items():
                column_name = f"che_weighted_{key}"  # Create prefixed column name
                result_df.at[index, column_name] = value

            # Add 'by_type' features if any exist
            for type_key, type_values in che_musique_stats['by_type'].items():
                column_name = f"che_bytype_{type_key}"
                result_df.at[index, column_name] = type_values
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
                        result_df.at[index, column_name] = value

                    # Add 'weighted' features
            for key, value in joc_musique_stats['weighted'].items():
                column_name = f"joc_weighted_{key}"  # Create prefixed column name
                result_df.at[index, column_name] = value

                # Add 'by_type' features if any exist
            for type_key, type_values in joc_musique_stats['by_type'].items():
                column_name = f"joc_bytype_{type_key}"
                result_df.at[index, column_name] = type_values

            # Phase 2: Calculate advanced derived features
            # Extract race-level information for Phase 2 features
            race_info = {
                'cheque': participant.get('cheque', 0),
                'partant': participant.get('partant', 0), 
                'dist': participant.get('dist', 0),
                'handicap_level_score': participant.get('handicap_level_score', 0)
            }
            
            phase2_features = Phase2FeatureCalculator.calculate_all_phase2_features(participant, race_info)
            for key, value in phase2_features.items():
                result_df.at[index, key] = value

        return result_df


def main():
    """
    Exemple d'utilisation du calculateur de features.
    """
    # Exemple avec des données de test
    test_participant = {
        'coursescheval': 10,
        'victoirescheval': 2,
        'placescheval': 5,
        'gainsCarriere': 100000,
        'musiqueche': '1 3 2 4 5',
        'nbCourseCouple': 5,
        'nbVictCouple': 1,
        'nbPlaceCouple': 3,
        'TxVictCouple': 20.0,
        'pourcVictCheval': 15.0,
        'pourcVictChevalHippo': 25.0,
        'pourcPlaceChevalHippo': 50.0,
        'pourcVictJockHippo': 30.0,
        'pourcPlaceJockHippo': 60.0
    }

    # Calculer les features
    features = FeatureCalculator.calculate_all_features(test_participant)

    # Afficher les résultats
    print("Features calculées:")
    for feature_name, value in features.items():
        print(f"{feature_name}: {value}")


if __name__ == "__main__":
    main()