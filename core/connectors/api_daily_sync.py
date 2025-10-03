import requests
import json
import sqlite3
import logging
import datetime
import pandas as pd
import os
import time
from typing import Dict, List, Union, Optional
from pathlib import Path

# Import the RaceDataConverter for preprocessing
from core.transformers.daily_race_transformer import RaceDataConverter

# Import AppConfig for environment configuration
from utils.env_setup import AppConfig


class RaceFetcher:
    """
    Module for fetching daily races from API, preprocessing features, and storing in the database.
    Utilizes RaceDataConverter to transform raw API data into model-ready features.
    Uses environment configuration from AppConfig.
    """

    def __init__(self,
                 db_name: str = None,
                 api_uid: str = "EhBCnKDamDfLkYJJjhWLf47xT7j1",
                 api_base_url: str = "https://aspiturf.com/api/api",
                 #api_base_url: str = "https://horseai.free.beeceptor.com/api",
                 verbose: bool = False):
        """
        Initialize the race fetcher.

        Args:
            db_name: Database name in config (default: active_db from config)
            api_uid: User ID for API authentication
            api_base_url: Base URL for the API
            verbose: Whether to output verbose logs
        """
        # Initialize config
        self.config = AppConfig()

        # Store verbose flag
        self.verbose = verbose

        # Get database name from config if not provided
        if db_name is None:
            # Use active_db from config
            self.db_name = self.config._config.base.active_db
        else:
            self.db_name = db_name

        # Get database path from config
        self.db_path = self.config.get_sqlite_dbpath(self.db_name)

        self.api_uid = api_uid
        self.api_base_url = api_base_url

        # Setup logging
        self._setup_logging()

        # Initialize the RaceDataConverter for preprocessing
        self.converter = RaceDataConverter(self.db_path)

        # Rate limiting for API calls
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # API endpoint configurations for multi-endpoint data collection
        self.endpoints = {
            'jour': '/jour',                    # Daily races (primary endpoint)
            'reunion': '/reunion',              # Meeting details
            'course': '/course',                # Race details
            'cheval': '/cheval',               # Horse career data
            'jockey': '/jockey',               # Jockey context
            'commentaires': '/commentaires',    # Race insights
            'comparaison': '/comparaison'       # Race comparison data
        }

        # Initialize database
        self._ensure_database()

        if self.verbose:
            self.logger.info(f"RaceFetcher initialized with database: {self.db_path} ({self.db_name})")

    # Also update _setup_logging
    def _setup_logging(self):
        """Set up logging with proper verbose control."""
        # Determine log directory from config
        root_dir = self.config._config.base.rootdir
        log_dir = os.path.join(root_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # Create log file path
        log_file = os.path.join(log_dir, f"race_fetcher_{self.db_name}.log")

        # Get or create logger
        self.logger = logging.getLogger("RaceFetcher")

        # Remove existing handlers
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)

        # Set level based on verbose flag
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Add file handler (always)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

        # Add console handler only if verbose
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)

    def _ensure_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create daily_race table with enhanced fields
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

            -- Enhanced race-level fields for competitive analysis
            cheque REAL,                        -- Race purse amount
            handi_raw VARCHAR(255),             -- Raw handicap classification
            reclam VARCHAR(50),                 -- Claiming race indicator
            sex VARCHAR(50),                    -- Sex restrictions
            tempscourse VARCHAR(100),           -- Race time information
            handicap_level_score REAL,          -- Derived handicap encoding

            participants JSON,                  -- Enhanced participants data with new fields
            raw_data JSON,                      -- Raw API data for the race
            processed_data JSON,                -- Processed feature data from transformer
            prediction_results JSON,            -- Prediction results (to be filled later)
            actual_results JSON,                -- Actual results (to be filled later)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
        ''')

        # Create an index on comp for faster lookups
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_daily_race_comp ON daily_race(comp)
        ''')

        # Create an index on jour for date-based lookups
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_daily_race_jour ON daily_race(jour)
        ''')

        # Add enhanced fields to existing table if they don't exist
        cursor.execute("PRAGMA table_info(daily_race)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        enhanced_fields = {
            'cheque': 'REAL',
            'handi_raw': 'VARCHAR(255)',
            'reclam': 'VARCHAR(50)',
            'sex': 'VARCHAR(50)',
            'tempscourse': 'VARCHAR(100)',
            'handicap_level_score': 'REAL'
        }

        for field_name, field_type in enhanced_fields.items():
            if field_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE daily_race ADD COLUMN {field_name} {field_type}")
                    self.logger.info(f"Added enhanced field: {field_name}")
                except Exception as e:
                    self.logger.warning(f"Could not add field {field_name}: {str(e)}")

        conn.commit()
        conn.close()
        self.logger.info("Database structure ensured")

    def _rate_limit(self):
        """Implement rate limiting to avoid overwhelming the API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_api_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make an API request with proper error handling and rate limiting.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            API response data or None if failed
        """
        self._rate_limit()

        # Add user ID to parameters
        params_with_uid = {'uid': self.api_uid, **params}

        url = f"{self.api_base_url}{endpoint}"

        try:
            if self.verbose:
                self.logger.info(f"Requesting {endpoint} with params: {params}")

            response = requests.get(url, params=params_with_uid, timeout=30)
            response.raise_for_status()

            data = response.json()

            if self.verbose:
                self.logger.info(f"✅ {endpoint} request successful: {len(data) if isinstance(data, list) else 1} records")

            return data

        except requests.RequestException as e:
            self.logger.error(f"❌ API request failed for {endpoint}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ JSON decode error for {endpoint}: {str(e)}")
            return None

    def _build_api_url(self, date: str) -> str:
        """
        Build API URL for a specific date.

        Args:
            date: Date string in format YYYY-MM-DD

        Returns:
            Complete API URL
        """
        return f"{self.api_base_url}?jour={date}&uid={self.api_uid}"
        #return "https://e7aaf339-293a-45ba-a477-c03d02480312.mock.pstmn.io"

    def fetch_races(self, date: str = None) -> Dict:
        """
        Fetch all races for a given date from the API using jour endpoint.

        Args:
            date: Date string in format YYYY-MM-DD (default: today)

        Returns:
            Dictionary with race data
        """
        # Use today's date if none provided
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")

        params = {'jour': date}
        data = self._make_api_request(self.endpoints['jour'], params)

        if data:
            self.logger.info(f"Successfully fetched races for {date}: {len(data)} entries")
            return data
        else:
            self.logger.error(f"Error fetching races for {date}")
            raise Exception(f"Failed to fetch races for {date}")

    def fetch_reunion_details(self, reunion_id: str, date: str) -> Optional[Dict]:
        """
        Fetch detailed meeting information for enhanced race context.

        Args:
            reunion_id: Meeting identifier
            date: Date string

        Returns:
            Meeting details or None if failed
        """
        params = {'reunion': reunion_id, 'jour': date}
        return self._make_api_request(self.endpoints['reunion'], params)

    def fetch_course_details(self, course_id: str) -> Optional[Dict]:
        """
        Fetch detailed race information.

        Args:
            course_id: Race identifier (comp)

        Returns:
            Race details or None if failed
        """
        params = {'course': course_id}
        return self._make_api_request(self.endpoints['course'], params)

    def fetch_horse_career(self, horse_id: str) -> Optional[Dict]:
        """
        Fetch comprehensive horse career data when insufficient data exists.

        Args:
            horse_id: Horse identifier

        Returns:
            Horse career data or None if failed
        """
        params = {'cheval': horse_id}
        return self._make_api_request(self.endpoints['cheval'], params)

    def fetch_jockey_context(self, jockey_id: str) -> Optional[Dict]:
        """
        Fetch jockey performance context when recent data missing.

        Args:
            jockey_id: Jockey identifier

        Returns:
            Jockey context data or None if failed
        """
        params = {'jockey': jockey_id}
        return self._make_api_request(self.endpoints['jockey'], params)

    def fetch_race_commentary(self, course_id: str) -> Optional[Dict]:
        """
        Fetch race commentary for major races.

        Args:
            course_id: Race identifier

        Returns:
            Race commentary or None if failed
        """
        params = {'course': course_id}
        return self._make_api_request(self.endpoints['commentaires'], params)

    def fetch_race_comparison(self, course_id: str) -> Optional[Dict]:
        """
        Fetch race comparison data for analytical insights.

        Args:
            course_id: Race identifier

        Returns:
            Race comparison data or None if failed
        """
        params = {'course': course_id}
        return self._make_api_request(self.endpoints['comparaison'], params)

    def _group_races_by_comp(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group race participants by race ID (comp).

        Args:
            data: API response data

        Returns:
            Dictionary mapping race IDs to lists of participants
        """
        races = {}

        # Process each participant
        for participant in data:
            # Get race ID from numcourse
            if 'numcourse' in participant and 'comp' in participant['numcourse']:
                comp = participant['numcourse']['comp']

                # Add race to dictionary if not already present
                if comp not in races:
                    races[comp] = []

                # Add participant to race
                races[comp].append(participant)

        return races

    def _calculate_handicap_encoding(self, handi_raw: str) -> float:
        """
        Calculate handicap level score from raw handicap classification.
        Consistent with training pipeline handicap encoding.

        Args:
            handi_raw: Raw handicap string from API

        Returns:
            Float representing handicap level score
        """
        if not handi_raw or handi_raw.strip() == '':
            return 0.0

        # Import handicap encoder for consistency with training
        try:
            from core.transformers.handicap_encoder import HandicapEncoder
            encoded_result = HandicapEncoder.parse_handicap_text(handi_raw)
            return float(encoded_result['handicap_level_score'])
        except ImportError:
            # Fallback simple encoding if encoder not available
            handi_str = str(handi_raw).upper().strip()

            # Simple mapping for common handicap levels
            simple_mapping = {
                'LISTE': 5.0,
                'GROUP': 4.5,
                'GROUPE': 4.5,
                'GRADED': 4.0,
                'HANDICAP': 3.0,
                'CLAIMING': 2.0,
                'MAIDEN': 1.0,
                'CONDITIONS': 3.5
            }

            for key, value in simple_mapping.items():
                if key in handi_str:
                    return value

            return 2.5  # Default middle value

    def _extract_race_info(self, participants: List[Dict]) -> Dict:
        """
        Extract race information from participants.

        Args:
            participants: List of participants

        Returns:
            Dictionary with race information
        """
        # Get race info from first participant's numcourse
        numcourse = participants[0].get('numcourse', {})

        # Check if we have arrival information (results)
        actual_results = None
        if 'arriv' in numcourse and numcourse['arriv']:
            # Just store the raw arrival data string
            actual_results = numcourse['arriv']
            self.logger.info(f"Found race results for {numcourse.get('comp')}: {actual_results}")
        else:
            # Race hasn't happened yet, just store "pending"
            actual_results = "pending"
            self.logger.info(f"Race {numcourse.get('comp')} hasn't happened yet")

        # Create race record for database with enhanced fields
        race_info = {
            'comp': numcourse.get('comp'),
            'jour': numcourse.get('jour'),
            'hippo': numcourse.get('hippo'),
            'reun': numcourse.get('reun'),
            'prix': numcourse.get('prix'),
            'prixnom': numcourse.get('prixnom'),
            'typec': numcourse.get('typec'),
            'partant': int(numcourse.get('partant')) if numcourse.get('partant') else None,
            'dist': int(numcourse.get('dist')) if numcourse.get('dist') else None,
            'handi': numcourse.get('handi'),
            'groupe': numcourse.get('groupe'),
            'quinte': bool(numcourse.get('quinte')),
            'natpis': numcourse.get('natpis'),
            'meteo': numcourse.get('meteo'),
            'corde': numcourse.get('corde'),
            'temperature': numcourse.get('temperature'),
            'forceVent': numcourse.get('forceVent'),
            'directionVent': numcourse.get('directionVent'),
            'nebulosite': numcourse.get('nebulositeLibelleCourt'),

            # Enhanced race-level fields for competitive analysis
            'cheque': float(numcourse.get('cheque', 0)) if numcourse.get('cheque') else None,
            'handi_raw': numcourse.get('handi'),  # Store raw handicap data
            'reclam': numcourse.get('reclam'),    # Claiming race indicator
            'sex': numcourse.get('sex'),          # Sex restrictions
            'tempscourse': numcourse.get('tempscourse'),  # Race time information
            'handicap_level_score': self._calculate_handicap_encoding(numcourse.get('handi')),

            'actual_results': actual_results
        }

        return race_info

    def store_race(self, race_info: Dict) -> int:
        """
        Store race information in the database.

        Args:
            race_info: Dictionary with race information

        Returns:
            Race ID in the database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if race already exists
        cursor.execute(
            "SELECT id FROM daily_race WHERE comp = ?",
            (race_info['comp'],)
        )
        existing = cursor.fetchone()

        # Make sure participants is available
        participants_json = race_info.get('participants', '[]')

        # Get actual_results
        actual_results = race_info.get('actual_results', json.dumps({"status": "pending"}))

        # Debug logging
        self.logger.info(f"Storing race {race_info['comp']} with {participants_json.count('{}')} participant records")

        if existing:
            # Update existing record with enhanced fields
            query = """
            UPDATE daily_race SET
                jour = ?, hippo = ?, reun = ?, prix = ?, prixnom = ?,
                typec = ?, partant = ?, dist = ?, handi = ?, groupe = ?,
                quinte = ?, natpis = ?, meteo = ?, corde = ?,
                temperature = ?, forceVent = ?, directionVent = ?,
                nebulosite = ?, cheque = ?, handi_raw = ?, reclam = ?,
                sex = ?, tempscourse = ?, handicap_level_score = ?,
                participants = ?, actual_results = ?, updated_at = ?
            WHERE comp = ?
            """
            cursor.execute(query, (
                race_info['jour'],
                race_info['hippo'],
                race_info['reun'],
                race_info['prix'],
                race_info['prixnom'],
                race_info['typec'],
                race_info['partant'],
                race_info['dist'],
                race_info['handi'],
                race_info['groupe'],
                race_info['quinte'],
                race_info['natpis'],
                race_info['meteo'],
                race_info['corde'],
                race_info['temperature'],
                race_info['forceVent'],
                race_info['directionVent'],
                race_info['nebulosite'],
                race_info['cheque'],
                race_info['handi_raw'],
                race_info['reclam'],
                race_info['sex'],
                race_info['tempscourse'],
                race_info['handicap_level_score'],
                participants_json,
                actual_results,
                datetime.datetime.now().isoformat(),
                race_info['comp']
            ))
            race_id = existing[0]
        else:
            # Insert new record with enhanced fields
            query = """
            INSERT INTO daily_race (
                comp, jour, hippo, reun, prix, prixnom, typec, partant,
                dist, handi, groupe, quinte, natpis, meteo, corde,
                temperature, forceVent, directionVent, nebulosite,
                cheque, handi_raw, reclam, sex, tempscourse, handicap_level_score,
                participants, actual_results, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (
                race_info['comp'],
                race_info['jour'],
                race_info['hippo'],
                race_info['reun'],
                race_info['prix'],
                race_info['prixnom'],
                race_info['typec'],
                race_info['partant'],
                race_info['dist'],
                race_info['handi'],
                race_info['groupe'],
                race_info['quinte'],
                race_info['natpis'],
                race_info['meteo'],
                race_info['corde'],
                race_info['temperature'],
                race_info['forceVent'],
                race_info['directionVent'],
                race_info['nebulosite'],
                race_info['cheque'],
                race_info['handi_raw'],
                race_info['reclam'],
                race_info['sex'],
                race_info['tempscourse'],
                race_info['handicap_level_score'],
                participants_json,
                actual_results,
                datetime.datetime.now().isoformat()
            ))
            race_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return race_id

    def process_race(self, participants: List[Dict]) -> Dict:
        """
        Process a race using the RaceDataConverter to generate preprocessed features.
        """
        try:
            # Extract race info (only pass participants)
            race_info = self._extract_race_info(participants)

            # Use RaceDataConverter to process the data
            processed_df = self.converter.process_race_data(participants)
            self.logger.info(f"Processed {len(processed_df)} participants with {len(processed_df.columns)} features")

            # Add processed data to race_info
            if len(processed_df) > 0:
                participants_json = processed_df.to_json(orient='records')
                race_info['participants'] = participants_json
            else:
                race_info['participants'] = '[]'  # Empty array if no participants

            return {
                'race_info': race_info,
                'processed_df': processed_df,
                'status': 'success'
            }
        except Exception as e:
            self.logger.error(f"Error processing race: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging

            # Extract basic race info without processed data
            race_info = self._extract_race_info(participants)
            race_info['participants'] = '[]'  # Empty JSON array

            return {
                'race_info': race_info,
                'processed_df': None,
                'status': 'error',
                'error': str(e)
            }

    def fetch_and_store_daily_races(self, date: str = None) -> Dict:
        """
        Main pipeline function that orchestrates the entire process:
        fetch > process > filter (is_in_france) > store

        Args:
            date: Date string in format YYYY-MM-DD (default: today)

        Returns:
            Dictionary with summary results
        """
        # Use today's date if none provided
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")

        try:
            # Step 1: Fetch races from API
            self.logger.info(f"Fetching races for {date}")
            api_data = self.fetch_races(date)

            # Step 2: Group participants by race
            grouped_races = self._group_races_by_comp(api_data)
            self.logger.info(f"Found {len(grouped_races)} races for {date}")

            # Step 3-5: Process, filter, and store each race
            results = self._process_filter_and_store_races(grouped_races, date)

            return results

        except Exception as e:
            self.logger.error(f"Error in fetch_and_store_daily_races: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            return {
                'date': date,
                'status': 'error',
                'error': str(e),
                'total_races': 0,
                'successful': 0,
                'failed': 0,
                'skipped': 0,
                'races': []
            }

    def _process_filter_and_store_races(self, grouped_races: Dict, date: str) -> Dict:
        """
        Process, filter, and store races.

        Args:
            grouped_races: Dictionary mapping race IDs to lists of participants
            date: Date string for the races

        Returns:
            Dictionary with processing results
        """
        results = []

        for comp, participants in grouped_races.items():
            try:
                # Step 3: Process the race
                self.logger.info(f"Processing race {comp}")
                processed_result = self.process_race(participants)

                if processed_result['status'] == 'error':
                    self.logger.warning(f"Failed to process race {comp}: {processed_result.get('error')}")
                    results.append({
                        'comp': comp,
                        'status': 'error',
                        'error': processed_result.get('error', 'Unknown processing error')
                    })
                    continue

                # Step 4: Filter - Check if race is in France
                hippo = processed_result['race_info'].get('hippo', 'Unknown')
                if not self.is_race_in_france(hippo):
                    self.logger.info(f"Race {comp} at {hippo} skipped as it is not in France")
                    results.append({
                        'comp': comp,
                        'status': 'skipped',
                        'reason': 'not_in_france',
                        'hippo': hippo
                    })
                    continue

                # Step 5: Store the race
                try:
                    race_id = self.store_race(processed_result['race_info'])

                    # Build result object
                    result = {
                        'race_id': race_id,
                        'comp': comp,
                        'hippo': hippo,
                        'prix': processed_result['race_info'].get('prix', 'Unknown'),
                        'partant': processed_result['race_info'].get('partant', 0),
                        'status': 'success'
                    }

                    # Add feature count if available
                    if processed_result['processed_df'] is not None:
                        result['feature_count'] = len(processed_result['processed_df'].columns)

                    results.append(result)
                    self.logger.info(f"Stored race {comp} with {result.get('feature_count', 0)} features")

                except Exception as e:
                    self.logger.error(f"Error storing race {comp}: {str(e)}")
                    results.append({
                        'comp': comp,
                        'status': 'error',
                        'error': f"Storage error: {str(e)}"
                    })

            except Exception as e:
                self.logger.error(f"Error processing race {comp}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

                results.append({
                    'comp': comp,
                    'status': 'error',
                    'error': str(e)
                })

        # Generate summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        skipped = sum(1 for r in results if r['status'] == 'skipped')

        summary = {
            'date': date,
            'total_races': len(results),
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'races': results
        }

        self.logger.info(f"Completed processing races for {date}: "
                         f"{successful} successful, {failed} failed, {skipped} skipped")

        return summary

    def _identify_data_gaps(self, participants: List[Dict]) -> Dict[str, str]:
        """
        Identify participants with insufficient data that need API enhancement.

        Args:
            participants: List of participant data

        Returns:
            Dict mapping participant IDs to gap types
        """
        gaps = {}

        for participant in participants:
            participant_id = participant.get('idche')
            if not participant_id:
                continue

            # Check for insufficient horse career data
            if self._has_insufficient_career_data(participant):
                gaps[participant_id] = 'horse_career'

            # Check for missing jockey context
            elif self._has_insufficient_jockey_data(participant):
                gaps[participant_id] = 'jockey_context'

        return gaps

    def _has_insufficient_career_data(self, participant: Dict) -> bool:
        """Check if horse career data is insufficient."""
        # Check for missing key career metrics
        career_fields = ['coursescheval', 'victoirescheval', 'placescheval', 'musiqueche']
        missing_fields = sum(1 for field in career_fields if not participant.get(field))

        return missing_fields >= 2  # Missing 2 or more key fields

    def _has_insufficient_jockey_data(self, participant: Dict) -> bool:
        """Check if jockey data is insufficient."""
        # Check for missing jockey performance data
        jockey_fields = ['musiquejoc', 'pourcVictJockHippo', 'pourcPlaceJockHippo']
        missing_fields = sum(1 for field in jockey_fields if not participant.get(field))

        return missing_fields >= 2  # Missing 2 or more key fields

    def _get_jockey_id_for_participant(self, participants: List[Dict], participant_id: str) -> Optional[str]:
        """Get jockey ID for a specific participant."""
        for participant in participants:
            if participant.get('idche') == participant_id:
                return participant.get('idJockey')
        return None

    def _is_major_race(self, race_info: Dict) -> bool:
        """
        Determine if a race is major enough to warrant commentary collection.

        Args:
            race_info: Race information dictionary

        Returns:
            True if race is considered major
        """
        # Major race indicators
        if race_info.get('quinte'):  # Quinte+ races
            return True

        if race_info.get('groupe') in ['1', '2', '3', 'I', 'II', 'III']:  # Group races
            return True

        # High-value races (check purse if available)
        cheque = race_info.get('cheque', 0)
        if isinstance(cheque, (int, float)) and cheque > 50000:  # €50,000+ purse
            return True

        return False

    def enhanced_race_collection(self, date: str) -> Dict:
        """
        Comprehensive race data collection using intelligent multi-endpoint strategy.

        Args:
            date: Date string in format YYYY-MM-DD

        Returns:
            Dict containing all collected data and collection summary
        """
        collection_start = time.time()

        if self.verbose:
            self.logger.info(f"🚀 Starting enhanced race collection for {date}")

        # Step 1: Primary data collection via jour endpoint
        primary_data = self.fetch_races(date)
        if not primary_data:
            return {
                'status': 'error',
                'error': 'Failed to fetch primary race data',
                'date': date,
                'api_calls': 1
            }

        # Group races by comp for processing
        races_by_comp = {}
        for participant in primary_data:
            if 'numcourse' in participant and 'comp' in participant['numcourse']:
                comp = participant['numcourse']['comp']
                if comp not in races_by_comp:
                    races_by_comp[comp] = []
                races_by_comp[comp].append(participant)

        # Step 2: Intelligent supplementary data collection
        enhanced_data = {}
        api_call_count = 1  # Started with 1 call for primary data

        for comp, participants in races_by_comp.items():
            enhanced_data[comp] = {
                'participants': participants,
                'supplementary_data': {}
            }

            race_info = participants[0].get('numcourse', {})

            # Collect meeting details for race context
            reunion_id = race_info.get('reun')
            if reunion_id:
                reunion_data = self.fetch_reunion_details(reunion_id, date)
                if reunion_data:
                    enhanced_data[comp]['supplementary_data']['reunion'] = reunion_data
                    api_call_count += 1

            # Collect detailed race information
            race_details = self.fetch_course_details(comp)
            if race_details:
                enhanced_data[comp]['supplementary_data']['course_details'] = race_details
                api_call_count += 1

            # Smart participant data enhancement
            participants_needing_enhancement = self._identify_data_gaps(participants)

            for participant_id, gap_type in participants_needing_enhancement.items():
                if gap_type == 'horse_career':
                    horse_data = self.fetch_horse_career(participant_id)
                    if horse_data:
                        enhanced_data[comp]['supplementary_data'][f'horse_{participant_id}'] = horse_data
                        api_call_count += 1

                elif gap_type == 'jockey_context':
                    jockey_id = self._get_jockey_id_for_participant(participants, participant_id)
                    if jockey_id:
                        jockey_data = self.fetch_jockey_context(jockey_id)
                        if jockey_data:
                            enhanced_data[comp]['supplementary_data'][f'jockey_{jockey_id}'] = jockey_data
                            api_call_count += 1

            # Collect commentary for major races only
            if self._is_major_race(race_info):
                commentary = self.fetch_race_commentary(comp)
                if commentary:
                    enhanced_data[comp]['supplementary_data']['commentary'] = commentary
                    api_call_count += 1

        collection_time = time.time() - collection_start

        summary = {
            'status': 'success',
            'date': date,
            'total_races': len(races_by_comp),
            'total_participants': len(primary_data),
            'api_calls': api_call_count,
            'collection_time_seconds': round(collection_time, 2),
            'enhanced_data': enhanced_data
        }

        if self.verbose:
            self.logger.info(f"✅ Enhanced collection complete: {api_call_count} API calls in {collection_time:.2f}s")

        return summary

    def get_collection_statistics(self, collection_result: Dict) -> Dict:
        """
        Generate detailed statistics from a collection result.

        Args:
            collection_result: Result from enhanced_race_collection

        Returns:
            Statistics dictionary
        """
        if collection_result.get('status') != 'success':
            return {'error': 'Collection was not successful'}

        enhanced_data = collection_result.get('enhanced_data', {})

        stats = {
            'races_collected': len(enhanced_data),
            'total_api_calls': collection_result.get('api_calls', 0),
            'api_efficiency': collection_result.get('total_participants', 0) / max(collection_result.get('api_calls', 1), 1),
            'supplementary_data_collected': {},
            'major_races_identified': 0
        }

        # Analyze supplementary data collected
        for comp, race_data in enhanced_data.items():
            supp_data = race_data.get('supplementary_data', {})

            for key in supp_data.keys():
                if key.startswith('horse_'):
                    stats['supplementary_data_collected']['horse_career_data'] = \
                        stats['supplementary_data_collected'].get('horse_career_data', 0) + 1
                elif key.startswith('jockey_'):
                    stats['supplementary_data_collected']['jockey_context_data'] = \
                        stats['supplementary_data_collected'].get('jockey_context_data', 0) + 1
                elif key == 'commentary':
                    stats['major_races_identified'] += 1
                elif key == 'reunion':
                    stats['supplementary_data_collected']['reunion_details'] = \
                        stats['supplementary_data_collected'].get('reunion_details', 0) + 1
                elif key == 'course_details':
                    stats['supplementary_data_collected']['course_details'] = \
                        stats['supplementary_data_collected'].get('course_details', 0) + 1

        return stats

    def fetch_and_store_enhanced_daily_races(self, date: str = None) -> Dict:
        """
        Enhanced daily race collection using multi-endpoint API strategy.
        Provides comprehensive data collection with intelligent API usage.

        Args:
            date: Date string in format YYYY-MM-DD (default: today)

        Returns:
            Dictionary with enhanced collection results
        """
        # Use today's date if none provided
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")

        try:
            # Step 1: Enhanced multi-endpoint data collection
            self.logger.info(f"Starting enhanced collection for {date}")
            collection_result = self.enhanced_race_collection(date)

            if collection_result['status'] != 'success':
                return collection_result

            # Step 2: Process enhanced data for each race
            enhanced_data = collection_result['enhanced_data']
            results = []

            for comp, race_data in enhanced_data.items():
                try:
                    # Extract participants with potential supplementary data integration
                    participants = race_data['participants']
                    supplementary_data = race_data.get('supplementary_data', {})

                    # Merge supplementary data into participants
                    enhanced_participants = self._merge_supplementary_data(
                        participants, supplementary_data
                    )

                    # Process the enhanced race data
                    processed_result = self.process_race(enhanced_participants)

                    if processed_result['status'] == 'error':
                        self.logger.warning(f"Failed to process enhanced race {comp}: {processed_result.get('error')}")
                        results.append({
                            'comp': comp,
                            'status': 'error',
                            'error': processed_result.get('error', 'Unknown processing error')
                        })
                        continue

                    # Filter - Check if race is in France
                    hippo = processed_result['race_info'].get('hippo', 'Unknown')
                    if not self.is_race_in_france(hippo):
                        self.logger.info(f"Enhanced race {comp} at {hippo} skipped as it is not in France")
                        results.append({
                            'comp': comp,
                            'status': 'skipped',
                            'reason': 'not_in_france',
                            'hippo': hippo
                        })
                        continue

                    # Store enhanced race data
                    race_id = self.store_race(processed_result['race_info'])

                    # Build result object with enhancement info
                    result = {
                        'race_id': race_id,
                        'comp': comp,
                        'hippo': hippo,
                        'prix': processed_result['race_info'].get('prix', 'Unknown'),
                        'partant': processed_result['race_info'].get('partant', 0),
                        'status': 'success',
                        'enhanced': True,
                        'supplementary_data_count': len(supplementary_data)
                    }

                    # Add feature count if available
                    if processed_result['processed_df'] is not None:
                        result['feature_count'] = len(processed_result['processed_df'].columns)

                    results.append(result)
                    self.logger.info(f"Stored enhanced race {comp} with {result.get('feature_count', 0)} features "
                                   f"and {len(supplementary_data)} supplementary data sources")

                except Exception as e:
                    self.logger.error(f"Error processing enhanced race {comp}: {str(e)}")
                    results.append({
                        'comp': comp,
                        'status': 'error',
                        'error': str(e)
                    })

            # Generate enhanced summary
            successful = sum(1 for r in results if r['status'] == 'success')
            failed = sum(1 for r in results if r['status'] == 'error')
            skipped = sum(1 for r in results if r['status'] == 'skipped')

            # Get collection statistics
            collection_stats = self.get_collection_statistics(collection_result)

            summary = {
                'date': date,
                'total_races': len(results),
                'successful': successful,
                'failed': failed,
                'skipped': skipped,
                'enhanced_collection': True,
                'api_calls_used': collection_result.get('api_calls', 0),
                'collection_time': collection_result.get('collection_time_seconds', 0),
                'collection_statistics': collection_stats,
                'races': results
            }

            self.logger.info(f"Completed enhanced processing races for {date}: "
                           f"{successful} successful, {failed} failed, {skipped} skipped "
                           f"using {collection_result.get('api_calls', 0)} API calls")

            return summary

        except Exception as e:
            self.logger.error(f"Error in enhanced daily race collection: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            return {
                'date': date,
                'status': 'error',
                'error': str(e),
                'enhanced_collection': False,
                'total_races': 0,
                'successful': 0,
                'failed': 0,
                'skipped': 0,
                'races': []
            }

    def _merge_supplementary_data(self, participants: List[Dict], supplementary_data: Dict) -> List[Dict]:
        """
        Merge supplementary API data into participant records for enhanced features.

        Args:
            participants: Original participant data
            supplementary_data: Additional data from multiple API endpoints

        Returns:
            Enhanced participant list with merged supplementary data
        """
        enhanced_participants = []

        for participant in participants:
            enhanced_participant = participant.copy()
            participant_id = participant.get('idche')
            jockey_id = participant.get('idJockey')

            # Merge horse career data if available
            if participant_id and f'horse_{participant_id}' in supplementary_data:
                horse_data = supplementary_data[f'horse_{participant_id}']
                enhanced_participant = self._merge_horse_career_data(enhanced_participant, horse_data)

            # Merge jockey context data if available
            if jockey_id and f'jockey_{jockey_id}' in supplementary_data:
                jockey_data = supplementary_data[f'jockey_{jockey_id}']
                enhanced_participant = self._merge_jockey_context_data(enhanced_participant, jockey_data)

            # Add race-level supplementary context
            if 'course_details' in supplementary_data:
                course_data = supplementary_data['course_details']
                enhanced_participant = self._merge_race_context_data(enhanced_participant, course_data)

            enhanced_participants.append(enhanced_participant)

        return enhanced_participants

    def _merge_horse_career_data(self, participant: Dict, horse_data: Dict) -> Dict:
        """Merge detailed horse career data into participant record."""
        enhanced = participant.copy()

        # Map career data fields (if missing in original data)
        career_mappings = {
            'totalRaces': 'coursescheval',
            'totalWins': 'victoirescheval',
            'totalPlaces': 'placescheval',
            'careerEarnings': 'gainsCarriere',
            'recentForm': 'musiqueche',
            'recordTime': 'recordG'
        }

        for api_field, std_field in career_mappings.items():
            if api_field in horse_data and (not enhanced.get(std_field) or enhanced.get(std_field) == ''):
                enhanced[std_field] = horse_data[api_field]

        return enhanced

    def _merge_jockey_context_data(self, participant: Dict, jockey_data: Dict) -> Dict:
        """Merge jockey performance context into participant record."""
        enhanced = participant.copy()

        # Map jockey data fields
        jockey_mappings = {
            'recentForm': 'musiquejoc',
            'winRate': 'pourcVictJockHippo',
            'placeRate': 'pourcPlaceJockHippo'
        }

        for api_field, std_field in jockey_mappings.items():
            if api_field in jockey_data and (not enhanced.get(std_field) or enhanced.get(std_field) == ''):
                enhanced[std_field] = jockey_data[api_field]

        return enhanced

    def _merge_race_context_data(self, participant: Dict, course_data: Dict) -> Dict:
        """Merge additional race context data into participant record."""
        enhanced = participant.copy()

        # Add race-level enhancements that might be missing
        race_mappings = {
            'trackRecord': 'trackRecord',
            'raceTime': 'tempscourse',
            'purseDetails': 'cheque'
        }

        for api_field, std_field in race_mappings.items():
            if api_field in course_data:
                enhanced[std_field] = course_data[api_field]

        return enhanced

    def is_race_in_france(self, hippo: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT Pays FROM hippodromes WHERE hippo = ?", (hippo,))
        result = cursor.fetchone()
        conn.close()

        if result and result[0] == 'France':
            return True
        else:
            return False

    def get_races_by_date(self, date: str) -> List[Dict]:
        """
        Get all races stored for a specific date.

        Args:
            date: Date string in format YYYY-MM-DD

        Returns:
            List of race information dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        # Get table info to check available columns
        cursor.execute("PRAGMA table_info(daily_race)")
        columns = [info[1] for info in cursor.fetchall()]  # Column names are in position 1

        # Base columns we always want if they exist
        base_cols = ["id", "comp", "jour", "hippo", "reun", "prix", "prixnom", "typec",
                     "partant", "dist", "quinte", "natpis","participants", "created_at"]

        # Filter to include only columns that exist
        select_cols = [col for col in base_cols if col in columns]

        # Add CASE statements for special indicator columns
        select_items = select_cols.copy()

        # Check for participants data
        if "participants" in columns:
            select_items.append(
                "CASE WHEN participants IS NOT NULL AND participants != '[]' THEN 1 ELSE 0 END as has_processed_data")
        else:
            select_items.append("0 as has_processed_data")

        # Check for predictions
        if "prediction_results" in columns:
            select_items.append("CASE WHEN prediction_results IS NOT NULL THEN 1 ELSE 0 END as has_predictions")
        else:
            select_items.append("0 as has_predictions")

        # Check for results
        if "actual_results" in columns:
            select_items.append(
                "CASE WHEN actual_results IS NOT NULL AND actual_results != 'pending' THEN 1 ELSE 0 END as has_results")
        else:
            select_items.append("0 as has_results")

        # Construct the query
        query = f"SELECT {', '.join(select_items)} FROM daily_race WHERE jour = ?"

        cursor.execute(query, (date,))
        rows = cursor.fetchall()

        # Convert rows to dictionaries
        races = [dict(row) for row in rows]
        conn.close()

        self.logger.info(f"Found {len(races)} races for date {date}")
        return races
    def get_all_daily_races(self) -> List[Dict]:
        """
        Get all races stored for a specific date.

        Args:
            date: Date string in format YYYY-MM-DD

        Returns:
            List of race information dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        # Get table info to check available columns
        cursor.execute("PRAGMA table_info(daily_race)")
        columns = [info[1] for info in cursor.fetchall()]  # Column names are in position 1

        # Base columns we always want if they exist
        base_cols = ["id", "comp", "jour", "hippo", "reun", "prix", "prixnom", "typec",
                     "partant", "dist", "quinte", "natpis","participants","prediction_results", "created_at"]

        # Filter to include only columns that exist
        select_cols = [col for col in base_cols if col in columns]

        # Add CASE statements for special indicator columns
        select_items = select_cols.copy()

        # Check for participants data
        if "participants" in columns:
            select_items.append(
                "CASE WHEN participants IS NOT NULL AND participants != '[]' THEN 1 ELSE 0 END as has_processed_data")
        else:
            select_items.append("0 as has_processed_data")

        # Check for predictions
        if "prediction_results" in columns:
            select_items.append("CASE WHEN prediction_results IS NOT NULL THEN 1 ELSE 0 END as has_predictions")
        else:
            select_items.append("0 as has_predictions")

        # Check for results
        if "actual_results" in columns:
            select_items.append(
                "CASE WHEN actual_results IS NOT NULL AND actual_results != 'pending' THEN 1 ELSE 0 END as has_results")
        else:
            select_items.append("0 as has_results")

        # Construct the query
        query = f"SELECT {', '.join(select_items)} FROM daily_race"

        cursor.execute(query)
        rows = cursor.fetchall()

        # Convert rows to dictionaries
        races = [dict(row) for row in rows]
        conn.close()

        self.logger.info(f"Found {len(races)} races")
        return races
    def get_race_by_comp(self, comp: str) -> Optional[Dict]:
        """
        Get race information by race ID (comp).

        Args:
            comp: Race identifier

        Returns:
            Race information dictionary or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get table info to check available columns
        cursor.execute("PRAGMA table_info(daily_race)")
        available_columns = [info[1] for info in cursor.fetchall()]  # Column names are in position 1

        # Execute query
        cursor.execute("SELECT * FROM daily_race WHERE comp = ?", (comp,))
        row = cursor.fetchone()

        if row:
            # Convert row to dict
            race_dict = dict(row)

            # Handle JSON fields that exist in the schema
            json_fields = ['participants', 'prediction_results', 'actual_results']
            for field in json_fields:
                if field in race_dict and race_dict[field]:
                    try:
                        race_dict[field] = json.loads(race_dict[field])
                    except json.JSONDecodeError:
                        # Keep as string if can't be parsed
                        pass
                    except Exception as e:
                        self.logger.warning(f"Error parsing {field} for race {comp}: {str(e)}")

            conn.close()
            return race_dict

        conn.close()
        return None

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
            (predictions_json, datetime.datetime.now().isoformat(), comp)
        )

        # Check if any row was updated
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()

        if success:
            self.logger.info(f"Updated prediction results for race {comp}")
        else:
            self.logger.warning(f"No race found with comp={comp} to update predictions")

        return success


# Command-line interface for easy scheduling
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch, process, and store daily races")
    parser.add_argument("--date", type=str, help="Date to fetch (YYYY-MM-DD format)")
    parser.add_argument("--db", type=str, help="Database name from config (defaults to active_db)")
    parser.add_argument("--list", action="store_true", help="List races for the specified date")
    parser.add_argument("--race", type=str, help="Get details for a specific race by ID")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    fetcher = RaceFetcher(db_name=args.db)

    if args.list:
        # List races for date
        date = args.date or datetime.datetime.now().strftime("%Y-%m-%d")
        races = fetcher.get_races_by_date(date)

        print(f"Races for {date}: {len(races)} found")
        for race in races:
            status = []
            if race['has_processed_data']:
                status.append("Processed")
            if race['has_predictions']:
                status.append("Predicted")
            if race['has_results']:
                status.append("Results")

            status_str = f" [{', '.join(status)}]" if status else ""
            print(f"{race['comp']}: {race['hippo']} - {race['prixnom']} - {race['partant']} runners{status_str}")

    elif args.race:
        # Get details for specific race
        race = fetcher.get_race_by_comp(args.race)

        if race:
            print(f"Race {race['comp']}:")
            print(f"  Venue: {race['hippo']}")
            print(f"  Race: {race['prix']} - {race['prixnom']}")
            print(f"  Type: {race['typec']} - Distance: {race['dist']}m")
            print(f"  Runners: {race['partant']}")
            print(f"  Date: {race['jour']}")
            print(f"  Created: {race['created_at']}")

            # Count participants
            if race['raw_data'] and isinstance(race['raw_data'], list):
                print(f"  Participants: {len(race['raw_data'])}")

            # Show feature stats
            if race['processed_data'] and isinstance(race['processed_data'], list):
                feature_count = len(race['processed_data'][0].keys()) if race['processed_data'] else 0
                print(f"  Processed features: {feature_count}")
            else:
                print("  No processed features available")

            # Show prediction status
            if race['prediction_results']:
                print("  Predictions available")

            # Show results status
            if race['actual_results']:
                print("  Actual results available")
        else:
            print(f"Race {args.race} not found")

    else:
        # Fetch, process and store races
        date = args.date or datetime.datetime.now().strftime("%Y-%m-%d")
        results = fetcher.fetch_and_store_daily_races(date)

        if results.get('status') == 'error':
            print(f"Error: {results.get('error')}")
        else:
            print(f"Fetched and stored races for {date}")
            print(f"Total races: {results['total_races']}")
            print(f"Successfully processed and stored: {results['successful']}")
            print(f"Failed: {results['failed']}")