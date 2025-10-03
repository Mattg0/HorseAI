import requests
import json
import logging
import time
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd

class EnhancedAPIManager:
    """
    Enhanced API Manager for comprehensive data collection using multiple endpoints.
    Implements intelligent API usage pattern to minimize costs while maximizing data richness.
    """

    def __init__(self,
                 api_uid: str = "EhBCnKDamDfLkYJJjhWLf47xT7j1",
                 api_base_url: str = "https://aspiturf.com/api/api",
                 verbose: bool = False):
        """
        Initialize the enhanced API manager.

        Args:
            api_uid: API user ID for authentication
            api_base_url: Base URL for the API
            verbose: Whether to output verbose logs
        """
        self.api_uid = api_uid
        self.api_base_url = api_base_url
        self.verbose = verbose

        # Setup logging
        self.logger = logging.getLogger("EnhancedAPIManager")
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # API endpoint configurations
        self.endpoints = {
            'jour': '/jour',                    # Daily races (primary endpoint)
            'reunion': '/reunion',              # Meeting details
            'course': '/course',                # Race details
            'cheval': '/cheval',               # Horse career data
            'jockey': '/jockey',               # Jockey context
            'commentaires': '/commentaires',    # Race insights
            'comparaison': '/comparaison'       # Race comparison data
        }

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

    def fetch_daily_races(self, date: str) -> Optional[List[Dict]]:
        """
        Fetch daily races using the primary jour endpoint.

        Args:
            date: Date string in format YYYY-MM-DD

        Returns:
            List of race participants or None if failed
        """
        params = {'jour': date}
        return self._make_api_request(self.endpoints['jour'], params)

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
        primary_data = self.fetch_daily_races(date)
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