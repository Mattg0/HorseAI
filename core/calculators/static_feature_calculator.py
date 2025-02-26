from typing import Dict, List, Union, Optional
import pandas
import numpy as np
import json
from core.calculators.musique_calculation import MusiqueFeatureExtractor

class FeatureCalculator:
    """
    Module de calcul des features dérivées pour les courses hippiques.
    Peut être utilisé à la fois pour l'entraînement et la prédiction.
    """

    @staticmethod
    def calculate_performance_ratios(participant: Dict) -> Dict[str, float]:
        """
        Calcule les ratios de performance du cheval.

        Args:
            participant: Dictionnaire contenant les données du participant

        Returns:
            Dict avec les ratios calculés
        """
        courses_total = participant.get('coursescheval', 0)

        if courses_total > 0:
            ratios = {
                'ratio_victoires': participant.get('victoirescheval,', 0) / courses_total,
                'ratio_places': participant.get('placescheval', 0) / courses_total,
                'gains_par_course': participant.get('gainsCarriere', 0) / courses_total
            }
        else:
            ratios = {
                'ratio_victoires': 0.0,
                'ratio_places': 0.0,
                'gains_par_course': 0.0
            }

        return ratios

    @staticmethod
    def calculate_couple_stats(participant: Dict) -> Dict[str, float]:
        """
        Calcule les statistiques du couple cheval/jockey.

        Args:
            participant: Dictionnaire contenant les données du participant

        Returns:
            Dict avec les statistiques calculées
        """
        couple_courses = participant.get('nbCourseCouple', 0)

        if couple_courses > 0:
            stats = {
                'efficacite_couple': participant.get('nbVictCouple', 0) / couple_courses,
                'regularite_couple': participant.get('nbPlaceCouple', 0) / couple_courses,
                'progression_couple': participant.get('TxVictCouple', 0) - participant.get('pourcVictCheval', 0)
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
                                         participant.get('pourcVictChevalHippo', 0) +
                                         participant.get('pourcPlaceChevalHippo', 0)
                                 ) / 2,
            'perf_jockey_hippo': (
                                         participant.get('pourcVictJockHippo', 0) +
                                         participant.get('pourcPlaceJockHippo', 0)
                                 ) / 2
        }

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

        # Itérer sur chaque ligne du DataFrame
        for index, participant_row in df.iterrows():
            # Extraire le participant comme un dictionnaire
            participant = participant_row.to_dict()

            # Calculer les ratios de performance
            ratios = FeatureCalculator.calculate_performance_ratios(participant)
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
            joc_musique_stats = jockey_musique_extractor.extract_features(participant['musiqueche'],
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
            for type_key, type_values in che_musique_stats['by_type'].items():
                column_name = f"che_bytype_{type_key}"
                result_df.at[index, column_name] = type_values

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