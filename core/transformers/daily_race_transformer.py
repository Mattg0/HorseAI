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

# Import unified data pipeline for consistency
from core.orchestrators.unified_data_pipeline import UnifiedDataPipeline



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

        # Initialize unified data pipeline for consistency
        self.unified_pipeline = UnifiedDataPipeline(verbose=False)

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
        #print(f"Database structure ensured at {self.db_path}")



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
        Standardize field names using a strict whitelist approach.

        Args:
            participant_data: Original participant dictionary

        Returns:
            Dictionary with only the allowed participant-specific fields
        """
        # Create an empty standardized dictionary
        standardized = {}

        # Define a strict whitelist of allowed fields
        # ONLY these fields will be included in the output
        allowed_fields = {
            'idche', 'cheval', 'cl', 'cotedirect', 'coteprob', 'numero',
            'handicapDistance', 'handicapPoids', 'poidmont', 'recence',
            'gainsAnneeEnCours', 'musiqueche', 'idJockey', 'musiquejoc',
            'idEntraineur', 'proprietaire', 'age', 'nbVictCouple', 'nbPlaceCouple',
            'victoirescheval', 'placescheval', 'TxVictCouple',
            'pourcVictChevalHippo', 'pourcPlaceChevalHippo',
            'pourcVictJockHippo', 'pourcPlaceJockHippo', 'coursescheval',

            # Enhanced participant-level fields for competitive analysis
            'derniereplace', 'dernierecote', 'dernierealloc', 'txreclam',
            'dernieredist', 'derniernbpartants', 'recordG', 'entraineur',
            'dernierEnt', 'tempstot', 'ecar', 'gainsCarriere',

            # Equipment and context fields
            'oeil', 'dernierOeil', 'oeilFirstTime', 'defoeil', 'defoeilPrec', 'defFirstTime',
            'vha', 'dernierTxreclam'
        }

        # Map of API field names to standardized names
        field_mapping = {
            'idChe': 'idche',
            'cote': 'cotedirect',
            'cotedirecte': 'cotedirect',
            'num': 'numero',
            'jockeid': 'idJockey',
            'entid': 'idEntraineur',
            'prop': 'proprietaire',

            # Enhanced field mappings
            'recordGains': 'recordG',
            'lastPlace': 'derniereplace',
            'lastOdds': 'dernierecote',
            'lastPurse': 'dernierealloc',
            'claimingTax': 'txreclam',
            'lastDistance': 'dernieredist',
            'lastFieldSize': 'derniernbpartants',
            'personalBest': 'recordG',
            'currentTrainer': 'entraineur',
            'previousTrainer': 'dernierEnt',
            'totalTime': 'tempstot',
            'margin': 'ecar',
            'careerEarnings': 'gainsCarriere'
        }

        # Copy only allowed fields from participant data
        for key, value in participant_data.items():
            # Skip the numcourse object entirely
            if key == 'numcourse':
                continue

            # Map the field name if needed
            mapped_key = field_mapping.get(key, key)

            # Only include whitelisted fields
            if mapped_key in allowed_fields:
                standardized[mapped_key] = value

        return standardized

    def convert_api_response(self, json_data: Union[str, List, Dict]) -> pd.DataFrame:
        """
        Convert API response to standardized DataFrame with strict field control.

        Args:
            json_data: API response in various formats

        Returns:
            DataFrame with only allowed participant fields
        """
        # Normalize to list of dictionaries
        raw_data = self._normalize_api_response(json_data)

        # Extract race type from first participant for musique processing
        race_typec = None
        if raw_data and 'numcourse' in raw_data[0] and 'typec' in raw_data[0]['numcourse']:
            race_typec = raw_data[0]['numcourse']['typec']
            print(f"Race type: {race_typec}")

        # Create standardized data with strict field control
        standardized_data = []
        for p in raw_data:
            # Apply strict whitelist filtering
            standardized = self._standardize_field_names(p)

            # Add typec temporarily for feature calculation - will be removed later
            if race_typec:
                standardized['typec'] = race_typec

            standardized_data.append(standardized)

        # Convert to DataFrame
        df = pd.DataFrame(standardized_data)

        # Apply static feature calculations using FeatureCalculator
        try:
            processed_df = self.feature_calculator.calculate_all_features(df)
            print(f"Applied static feature calculations on {len(processed_df)} participants")

            # Remove typec from final result if it was temporarily added
            #if 'typec' in processed_df.columns and race_typec:
            #    processed_df = processed_df.drop(columns=['typec'])

        except Exception as e:
            print(f"Error applying static feature calculations: {str(e)}")
            processed_df = df

        # Convert numeric fields to ensure proper types
        numeric_fields = [
            'age', 'cotedirect', 'coteprob', 'numero',
            'pourcVictChevalHippo', 'pourcPlaceChevalHippo',
            'pourcVictJockHippo', 'pourcPlaceJockHippo',
            'victoirescheval', 'placescheval', 'coursescheval',
            'nbVictCouple', 'nbPlaceCouple', 'TxVictCouple',
            'handicapDistance', 'handicapPoids', 'recence',

            # Enhanced numeric fields
            'derniereplace', 'dernierecote', 'dernierealloc', 'txreclam',
            'dernieredist', 'derniernbpartants', 'tempstot', 'ecar',
            'gainsCarriere', 'vha', 'dernierTxreclam'
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
            'perf_jockey_hippo': 0,

            # Enhanced field defaults
            'derniereplace': 10,          # Last race position (default middle field)
            'dernierecote': 10.0,         # Last race odds
            'dernierealloc': 0,           # Previous race purse
            'txreclam': 0,                # Claiming price
            'dernieredist': 2000,         # Last race distance (default)
            'derniernbpartants': 10,      # Last race field size
            'tempstot': 0,                # Individual finish time
            'ecar': 0,                    # Victory/defeat margin
            'gainsCarriere': 0,           # Career earnings
            'vha': 0,                     # VHA rating
            'dernierTxreclam': 0          # Previous claiming price
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

        # Special handling for recordG time format conversion
        if 'recordG' in result_df.columns:
            result_df['recordG'] = result_df['recordG'].apply(self._parse_race_time)

        return result_df

    def _parse_race_time(self, time_str):
        """
        Parse race time from format "1'29"4" to milliseconds.

        Args:
            time_str: Time string in various formats

        Returns:
            Float representing time in milliseconds, or 0 if invalid
        """
        if pd.isna(time_str) or not time_str:
            return 0.0

        try:
            time_str = str(time_str).strip()
            if not time_str:
                return 0.0

            # Handle format like "1'29"4" (1 minute, 29.4 seconds)
            if "'" in time_str and '"' in time_str:
                # Split by apostrophe to get minutes and seconds part
                parts = time_str.split("'")
                if len(parts) == 2:
                    minutes = int(parts[0])
                    # Remove quotes and parse seconds
                    seconds_part = parts[1].replace('"', '').replace('"', '')
                    seconds = float(seconds_part)

                    # Convert to milliseconds
                    total_milliseconds = (minutes * 60 + seconds) * 1000
                    return total_milliseconds

            # Handle format like "89.2" (just seconds)
            elif '.' in time_str:
                seconds = float(time_str)
                return seconds * 1000

            # Handle format like "89" (just seconds as integer)
            else:
                seconds = float(time_str)
                return seconds * 1000

        except (ValueError, IndexError):
            return 0.0



    def process_race_data(self, json_data: Union[str, List, Dict], race_context: Optional[Dict] = None) -> pd.DataFrame:
        """
        Process race data using the unified pipeline for training-prediction consistency.

        Args:
            json_data: JSON data from API
            race_context: Additional race-level context information

        Returns:
            DataFrame with standard features, matching historical data format
        """
        # First convert to basic DataFrame with standardized field names
        df = self.convert_api_response(json_data)

        # Extract race context from the data if not provided
        if race_context is None and len(df) > 0:
            # Try to extract race context from the first participant
            first_participant = df.iloc[0].to_dict()
            race_context = {
                'cheque': first_participant.get('cheque', 0),
                'partant': first_participant.get('partant', 0),
                'dist': first_participant.get('dist', 0),
                'handi': first_participant.get('handi', ''),
                'typec': first_participant.get('typec', 'Plat'),
                'groupe': first_participant.get('groupe', ''),
                'quinte': first_participant.get('quinte', False)
            }

        # Apply the unified data pipeline for consistency
        try:
            processed_df = self.unified_pipeline.process_race_data(
                df,
                race_context=race_context,
                source="daily_sync"
            )
            print(f"Processed {len(processed_df)} participants with {len(processed_df.columns)} features via unified pipeline")
        except Exception as e:
            print(f"Error applying unified pipeline: {str(e)}")
            # Fallback to basic feature calculation
            processed_df = self.feature_calculator.calculate_all_features(df)

        return processed_df


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

