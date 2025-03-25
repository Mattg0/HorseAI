import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Any, Optional

# Import existing calculators and extractors
from utils.env_setup import AppConfig
from core.calculators.static_feature_calculator import FeatureCalculator
from core.calculators.musique_calculation import MusiqueFeatureExtractor


class RaceDataConverter:
    """
    Enhanced converter for horse race API data that transforms API responses
    into a format suitable for model prediction using the same preprocessing
    and calculation methods used during model training.
    """

    def __init__(self, db_path: str):
        """
        Initialize the race data converter.

        Args:
            db_path: Path to SQLite database for storing races
        """
        self.db_path = db_path

        # Initialize config
        self.config = AppConfig()

        # Initialize calculators
        self.feature_calculator = FeatureCalculator()
        self.musique_extractor = MusiqueFeatureExtractor()

        # Create necessary database structure
        self._ensure_database()

    def _ensure_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create daily_race table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_race (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comp VARCHAR(50) NOT NULL,          -- Race identifier
            jour DATE NOT NULL,                 -- Race date
            hippo VARCHAR(100) NOT NULL,        -- Racecourse
            reun VARCHAR(20),                   -- Meeting number
            prix VARCHAR(20),                   -- Race number
            prixnom VARCHAR(255),               -- Race name
            typec VARCHAR(20),                  -- Race type (Plat, etc.)
            partant INTEGER,                    -- Number of runners
            dist INTEGER,                       -- Distance in meters
            handi VARCHAR(255),                 -- Handicap info
            groupe VARCHAR(50),                 -- Race class
            quinte BOOLEAN,                     -- Whether it's a Quinte+ race
            natpis VARCHAR(50),                 -- Track surface
            meteo VARCHAR(100),                 -- Weather
            corde VARCHAR(50),                  -- Rail position
            temperature REAL,                   -- Temperature
            forceVent REAL,                     -- Wind force
            directionVent VARCHAR(50),          -- Wind direction
            nebulosite VARCHAR(100),            -- Cloud cover
            participants JSON,                  -- Raw participants data
            processed_data JSON,                -- Processed data with calculated features
            prediction_results JSON,            -- Prediction results
            actual_results JSON,                -- Actual results
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
        ''')

        conn.commit()
        conn.close()
        print(f"Database structure ensured at {self.db_path}")

    def _extract_race_info(self, data: List[Dict]) -> Dict:
        """Extract race information from the first participant."""
        first_participant = data[0]
        race_info = first_participant.get('numcourse', {})

        # Create race record for database
        return {
            'comp': race_info.get('comp'),
            'jour': race_info.get('jour'),
            'hippo': race_info.get('hippo'),
            'reun': race_info.get('reun'),
            'prix': race_info.get('prix'),
            'prixnom': race_info.get('prixnom'),
            'typec': race_info.get('typec'),
            'partant': int(race_info.get('partant')) if race_info.get('partant') else None,
            'dist': int(race_info.get('dist')) if race_info.get('dist') else None,
            'handi': race_info.get('handi'),
            'groupe': race_info.get('groupe'),
            'quinte': bool(race_info.get('quinte')),
            'natpis': race_info.get('natpis'),
            'meteo': race_info.get('meteo'),
            'corde': race_info.get('corde'),
            'temperature': race_info.get('temperature'),
            'forceVent': race_info.get('forceVent'),
            'directionVent': race_info.get('directionVent'),
            'nebulosite': race_info.get('nebulositeLibelleCourt'),
            'participants': json.dumps(data)
        }

    def _normalize_api_response(self, json_data: Union[str, List, Dict]) -> List[Dict]:
        """
        Normalize API response into a standard format.

        Args:
            json_data: API response in various formats

        Returns:
            List of participant dictionaries
        """
        # Convert to list of dictionaries if needed
        if isinstance(json_data, str):
            data = json.loads(json_data)
        elif isinstance(json_data, dict):
            # Single participant or dict with 'participants' key
            if 'participants' in json_data:
                data = json_data['participants']
            else:
                data = [json_data]
        else:
            data = json_data

        return data

    def _standardize_field_names(self, participant_data: Dict) -> Dict:
        """
        Standardize field names to be consistent with training data.

        Args:
            participant_data: Original participant dictionary

        Returns:
            Dictionary with standardized field names
        """
        # Get numcourse data (race info)
        numcourse = participant_data.get('numcourse', {})

        # Map of API field names to standardized names
        field_mapping = {
            'idChe': 'idche',
            'musiqueche': 'musiqueche',
            'musiquejoc': 'musiquejoc',
            'gainsCarriere': 'gainsCarriere',
            'gainsAnneeEnCours': 'gainsAnneeEnCours',
        }

        # Create standardized data
        standardized = {}

        # Copy existing fields
        for key, value in participant_data.items():
            # Use mapped name if exists, otherwise keep original
            new_key = field_mapping.get(key, key)
            standardized[new_key] = value

        # Add race fields from numcourse if not already present
        if numcourse:
            race_fields = ['comp', 'hippo', 'typec', 'dist', 'temperature',
                           'natpis', 'meteo', 'forceVent', 'directionVent',
                           'nebulositeLibelleCourt']

            for field in race_fields:
                if field not in standardized and field in numcourse:
                    standardized[field] = numcourse[field]

        return standardized

    def convert_api_response(self, json_data: Union[str, List, Dict]) -> pd.DataFrame:

        # Normalize to list of dictionaries
        raw_data = self._normalize_api_response(json_data)

        # Standardize field names for each participant
        standardized_data = [self._standardize_field_names(p) for p in raw_data]

        # Convert to DataFrame
        df = pd.DataFrame(standardized_data)

        # Apply static feature calculations using FeatureCalculator
        try:
            processed_df = self.feature_calculator.calculate_all_features(df)
            print(f"Applied static feature calculations on {len(processed_df)} participants")
        except Exception as e:
            print(f"Error applying static feature calculations: {str(e)}")
            processed_df = df

            # Convert numeric fields
        numeric_fields = [
            'age', 'cotedirect', 'coteprob', 'pourcVictChevalHippo',
            'pourcPlaceChevalHippo', 'pourcVictJockHippo', 'pourcPlaceJockHippo',
            'victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere',
            'gainsAnneeEnCours', 'nbCourseCouple', 'nbVictCouple', 'nbPlaceCouple',
            'TxVictCouple', 'recence', 'dist', 'temperature', 'forceVent'
        ]

        for field in numeric_fields:
            if field in processed_df.columns:
                processed_df[field] = pd.to_numeric(processed_df[field], errors='coerce')

            # Handle missing values for important features
        processed_df = self._handle_missing_values(processed_df)

        return processed_df


def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Args:
        df: DataFrame with possibly missing values

    Returns:
        DataFrame with handled missing values
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # Essential numeric fields that should have defaults
    numeric_defaults = {
        'age': 5,
        'victoirescheval': 0,
        'placescheval': 0,
        'coursescheval': 0,
        'pourcVictChevalHippo': 0,
        'pourcPlaceChevalHippo': 0,
        'pourcVictJockHippo': 0,
        'pourcPlaceJockHippo': 0,
        'nbCourseCouple': 0,
        'nbVictCouple': 0,
        'nbPlaceCouple': 0,
        'TxVictCouple': 0,
        'ratio_victoires': 0,
        'ratio_places': 0,
        'gains_par_course': 0,
        'efficacite_couple': 0,
        'regularite_couple': 0,
        'progression_couple': 0,
        'perf_cheval_hippo': 0,
        'perf_jockey_hippo': 0
    }

    # Fill missing values with defaults
    for field, default in numeric_defaults.items():
        if field in result_df.columns:
            result_df[field] = result_df[field].fillna(default)

    # Special handling for musique (performance history)
    if 'musiqueche' in result_df.columns:
        result_df['musiqueche'] = result_df['musiqueche'].fillna('')

    if 'musiquejoc' in result_df.columns:
        result_df['musiquejoc'] = result_df['musiquejoc'].fillna('')

    return result_df


def _extract_musique_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from musique (performance history) strings.

    Args:
        df: DataFrame with musique columns

    Returns:
        DataFrame with additional musique-derived features
    """
    result_df = df.copy()

    # Process horse musique if available
    if 'musiqueche' in df.columns and 'typec' in df.columns:
        for index, row in df.iterrows():
            try:
                musique = row['musiqueche']
                race_type = row['typec']

                if not pd.isna(musique) and musique:
                    # Extract features using MusiqueFeatureExtractor
                    features = self.musique_extractor.extract_features(musique, race_type)

                    # Add global features with prefix
                    for key, value in features['global'].items():
                        column_name = f"che_global_{key}"
                        result_df.at[index, column_name] = value

                    # Add weighted features with prefix
                    for key, value in features['weighted'].items():
                        column_name = f"che_weighted_{key}"
                        result_df.at[index, column_name] = value

                    # Add by_type features if available
                    for type_key, type_values in features['by_type'].items():
                        column_name = f"che_bytype_{type_key}"
                        result_df.at[index, column_name] = type_values
            except Exception as e:
                print(f"Error processing musique for index {index}: {str(e)}")

    # Process jockey musique if available
    if 'musiquejoc' in df.columns and 'typec' in df.columns:
        for index, row in df.iterrows():
            try:
                musique = row['musiquejoc']
                race_type = row['typec']

                if not pd.isna(musique) and musique:
                    # Extract features using MusiqueFeatureExtractor
                    features = self.musique_extractor.extract_features(musique, race_type)

                    # Add global features with prefix
                    for key, value in features['global'].items():
                        column_name = f"joc_global_{key}"
                        result_df.at[index, column_name] = value

                    # Add weighted features with prefix
                    for key, value in features['weighted'].items():
                        column_name = f"joc_weighted_{key}"
                        result_df.at[index, column_name] = value

                    # Add by_type features if available
                    for type_key, type_values in features['by_type'].items():
                        column_name = f"joc_bytype_{type_key}"
                        result_df.at[index, column_name] = type_values
            except Exception as e:
                print(f"Error processing jockey musique for index {index}: {str(e)}")

    return result_df


def _apply_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply additional features that might be relevant for prediction.

    Args:
        df: DataFrame with basic features

    Returns:
        DataFrame with additional calculated features
    """
    result_df = df.copy()

    # Extract musique features
    result_df = self._extract_musique_features(result_df)

    # Calculate derived features not handled by FeatureCalculator
    # Add weight-based features if poidmont exists
    if 'poidmont' in result_df.columns:
        # Convert to numeric, coercing errors to NaN
        result_df['poidmont'] = pd.to_numeric(result_df['poidmont'], errors='coerce')

        # Calculate weight-related features
        if not result_df['poidmont'].isna().all():
            result_df['weight_normalized'] = (result_df['poidmont'] - result_df['poidmont'].min()) / \
                                             (result_df['poidmont'].max() - result_df['poidmont'].min())

            result_df['weight_from_mean'] = result_df['poidmont'] - result_df['poidmont'].mean()

    # Calculate recency features if recence exists
    if 'recence' in result_df.columns:
        result_df['recence'] = pd.to_numeric(result_df['recence'], errors='coerce').fillna(100)
        result_df['recency_factor'] = 1 / (1 + result_df['recence'] / 30)  # Normalize with 30-day scale

    # Add position draw advantage/disadvantage if corde exists
    if 'corde' in result_df.columns and 'partant' in result_df.columns:
        # Convert to numeric
        result_df['corde'] = pd.to_numeric(result_df['corde'], errors='coerce')
        result_df['partant'] = pd.to_numeric(result_df['partant'], errors='coerce')

        # Calculate normalized position (0 to 1)
        valid_mask = ~result_df['corde'].isna() & ~result_df['partant'].isna() & (result_df['partant'] > 0)
        result_df.loc[valid_mask, 'corde_normalized'] = (result_df.loc[valid_mask, 'corde'] - 1) / \
                                                        (result_df.loc[valid_mask, 'partant'] - 1)

        # Calculate inside/outside position advantage
        result_df['inside_position'] = 1 - result_df.get('corde_normalized', 1)
        result_df['outside_position'] = result_df.get('corde_normalized', 0)

    return result_df


def process_race_data(self, json_data: Union[str, List, Dict]) -> pd.DataFrame:
    """
    Process race data with full feature calculation.

    Args:
        json_data: JSON data from API

    Returns:
        DataFrame with complete set of features needed for prediction
    """
    # First convert to basic DataFrame
    df = self.convert_api_response(json_data)

    # Apply additional feature calculations
    processed_df = self._apply_additional_features(df)

    print(f"Processed {len(processed_df)} participants with {len(processed_df.columns)} features")

    return processed_df


def store_race(self, race_data: Dict, processed_df: Optional[pd.DataFrame] = None) -> int:
    """
    Store race data in the database.

    Args:
        race_data: Dictionary with race information
        processed_df: Optional DataFrame with processed features

    Returns:
        Race ID in the database
    """
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Convert processed DataFrame to JSON if provided
    processed_json = None
    if processed_df is not None:
        processed_json = processed_df.to_json(orient='records')

    # Check if race already exists
    cursor.execute(
        "SELECT id FROM daily_race WHERE comp = ?",
        (race_data['comp'],)
    )
    existing = cursor.fetchone()

    if existing:
        # Update existing record
        query = """
            UPDATE daily_race SET
                jour = ?, hippo = ?, reun = ?, prix = ?, prixnom = ?,
                typec = ?, partant = ?, dist = ?, handi = ?, groupe = ?,
                quinte = ?, natpis = ?, meteo = ?, corde = ?,
                temperature = ?, forceVent = ?, directionVent = ?,
                nebulosite = ?, participants = ?, processed_data = ?, 
                updated_at = ?
            WHERE comp = ?
            """
        cursor.execute(query, (
            race_data['jour'],
            race_data['hippo'],
            race_data['reun'],
            race_data['prix'],
            race_data['prixnom'],
            race_data['typec'],
            race_data['partant'],
            race_data['dist'],
            race_data['handi'],
            race_data['groupe'],
            race_data['quinte'],
            race_data['natpis'],
            race_data['meteo'],
            race_data['corde'],
            race_data['temperature'],
            race_data['forceVent'],
            race_data['directionVent'],
            race_data['nebulosite'],
            race_data['participants'],
            processed_json,
            datetime.now().isoformat(),
            race_data['comp']
        ))
        race_id = existing[0]
    else:
        # Insert new record
        query = """
            INSERT INTO daily_race (
                comp, jour, hippo, reun, prix, prixnom, typec, partant,
                dist, handi, groupe, quinte, natpis, meteo, corde,
                temperature, forceVent, directionVent, nebulosite,
                participants, processed_data, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        cursor.execute(query, (
            race_data['comp'],
            race_data['jour'],
            race_data['hippo'],
            race_data['reun'],
            race_data['prix'],
            race_data['prixnom'],
            race_data['typec'],
            race_data['partant'],
            race_data['dist'],
            race_data['handi'],
            race_data['groupe'],
            race_data['quinte'],
            race_data['natpis'],
            race_data['meteo'],
            race_data['corde'],
            race_data['temperature'],
            race_data['forceVent'],
            race_data['directionVent'],
            race_data['nebulosite'],
            race_data['participants'],
            processed_json,
            datetime.now().isoformat()
        ))
        race_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return race_id


def convert_and_store(self, json_data: Union[str, List, Dict]) -> Dict:
    """
    Convert API data, process features, and store in the database.

    Args:
        json_data: JSON data from API

    Returns:
        dict: Race information including processed data
    """
    # Convert input to list if needed
    data = self._normalize_api_response(json_data)

    # Extract race info for storage
    race_info = self._extract_race_info(data)

    # Process data with full feature calculation
    processed_df = self.process_race_data(data)

    # Store in database
    race_id = self.store_race(race_info, processed_df)

    return {
        'race_id': race_id,
        'comp': race_info['comp'],
        'race_info': {
            'hippo': race_info['hippo'],
            'jour': race_info['jour'],
            'prix': race_info['prix'],
            'prixnom': race_info['prixnom'],
            'dist': race_info['dist'],
            'typec': race_info['typec']
        },
        'data_frame': processed_df
    }


def update_prediction_results(self, comp: str, prediction_results: Union[pd.DataFrame, List[Dict], str]) -> bool:
    """
    Update the prediction results for a race.

    Args:
        comp: Race identifier
        prediction_results: Prediction results as DataFrame, list of dicts, or JSON string

    Returns:
        bool: Success status
    """
    # Convert prediction results to JSON string if necessary
    if isinstance(prediction_results, pd.DataFrame):
        predictions_json = prediction_results.to_json(orient='records')
    elif isinstance(prediction_results, list):
        predictions_json = json.dumps(prediction_results)
    else:
        predictions_json = prediction_results

    # Update database
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE daily_race SET prediction_results = ?, updated_at = ? WHERE comp = ?",
        (predictions_json, datetime.now().isoformat(), comp)
    )

    # Check if any row was updated
    success = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return success


def update_actual_results(self, comp: str, results: List[Dict]) -> bool:
    """
    Update the actual race results.

    Args:
        comp: Race identifier
        results: Actual results list [{"numero": 1, "position": 3}, ...]

    Returns:
        bool: Success status
    """
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Format the results for consistency
    formatted_results = []
    for result in results:
        # Ensure each result has at least numero and position
        if 'numero' in result and 'position' in result:
            formatted_results.append({
                'numero': result['numero'],
                'position': result['position'],
                'cheval': result.get('cheval'),
                'jockey': result.get('jockey')
            })

    # Update the database
    cursor.execute(
        "UPDATE daily_race SET actual_results = ?, updated_at = ? WHERE comp = ?",
        (json.dumps(formatted_results), datetime.now().isoformat(), comp)
    )

    # Check if any row was updated
    success = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return success


def get_race_by_comp(self, comp: str) -> Optional[Dict]:
    """
    Retrieve a race from the database by its comp identifier.

    Args:
        comp: Race identifier

    Returns:
        Dictionary with race data or None if not found
    """
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM daily_race WHERE comp = ?",
        (comp,)
    )

    row = cursor.fetchone()

    if row:
        # Convert row to dict
        race_dict = dict(row)

        # Parse JSON fields
        for field in ['participants', 'processed_data', 'prediction_results', 'actual_results']:
            if race_dict[field]:
                try:
                    race_dict[field] = json.loads(race_dict[field])
                except json.JSONDecodeError:
                    race_dict[field] = None

        conn.close()
        return race_dict

    conn.close()
    return None

