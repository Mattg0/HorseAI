# data_transformers/race_transformer.py

import pandas as pd
from decimal import Decimal
from core.calculators.static_feature_calculator import FeatureCalculator


def transform_race_data(df):
    """
    Transform the fetched data into course info and participants.

    Args:
        df: DataFrame with raw race data

    Returns:
        Tuple of (course_infos, participants) DataFrames
    """
    features = FeatureCalculator.calculate_all_features(df)

    # Extract course info columns
    course_infos_columns = [
        'jour', 'quinte', 'hippo', 'reun', 'prix', 'partant', 'meteo',
        'dist', 'corde', 'natpis', 'pistegp', 'typec', 'temperature',
        'forceVent', 'directionVent', 'nebulositeLibelleCourt', 'handi',
        # Phase 2: Additional race-level fields
        'cheque', 'reclam', 'groupe', 'sex', 'tempscourse'
    ]

    # Create course info DataFrame
    course_infos = df.loc[:, df.columns.isin(course_infos_columns + ['id'])]
    course_infos = course_infos.drop_duplicates(subset=['id'])

    # Create participants DataFrame
    participants = features.drop(columns=[
        col for col in features.columns if col in course_infos_columns
    ])

    return course_infos, participants


def convert_decimal(value):
    """
    Convert Decimal objects to float for JSON serialization.

    Args:
        value: Value to convert

    Returns:
        Float if value is Decimal, otherwise original value
    """
    if isinstance(value, Decimal):
        return float(value)
    return value


# data_transformers/result_transformer.py

import json
from collections import defaultdict


def transform_results(df_raw_data):
    """
    Transform raw data into race results format.

    Args:
        df_raw_data: DataFrame with raw race data

    Returns:
        DataFrame with race results
    """
    course_results = defaultdict(list)

    # Process each row of data
    for index, row in df_raw_data.iterrows():
        comp = row.get('id')  # Course ID
        cl = row.get('cl')  # Finish position
        numero = int(row.get('numero'))  # Horse number
        idche = int(row.get('idche'))  # Horse ID

        # Add result to list for this race
        course_results[comp].append({
            'narrivee': cl,
            'cheval': numero,
            'idche': idche
        })

    # Serialize results for each race
    serialized_results = []
    for comp, results in course_results.items():
        # Sort results by arrival order (cl)
        results.sort(key=lambda x: (isinstance(x['narrivee'], str), x['narrivee']))

        serialized_results.append({
            'comp': comp,
            'ordre_arrivee': json.dumps(results)  # Serialize to JSON
        })

    return pd.DataFrame(serialized_results)