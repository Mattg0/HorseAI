import pandas as pd
import json
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np

# Import components for validation
from core.connectors.api_daily_sync import RaceFetcher
from core.orchestrators.unified_data_pipeline import UnifiedDataPipeline


class SyncSystemValidator:
    """
    Comprehensive validation system for the enhanced daily API sync.
    Ensures data quality, consistency, and system reliability.
    """

    def __init__(self, db_name: str = None, verbose: bool = True):
        """
        Initialize the validation system.

        Args:
            db_name: Database name to validate against
            verbose: Whether to output detailed validation logs
        """
        self.verbose = verbose
        self.db_name = db_name

        # Setup logging
        self.logger = logging.getLogger("SyncSystemValidator")
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

        # Initialize components
        self.race_fetcher = RaceFetcher(db_name=db_name, verbose=verbose)
        self.unified_pipeline = UnifiedDataPipeline(verbose=verbose)

        if self.verbose:
            self.logger.info("✅ Sync system validator initialized")

    def validate_enhanced_sync(self, date: str = None) -> Dict:
        """
        Comprehensive validation of the enhanced sync system.

        Args:
            date: Date to validate (default: yesterday for completed races)

        Returns:
            Validation report dictionary
        """
        if date is None:
            # Use yesterday to ensure races are completed
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        if self.verbose:
            self.logger.info(f"🔍 Starting comprehensive validation for {date}")

        validation_report = {
            'date': date,
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'pass',
            'validations': {}
        }

        # 1. Database Schema Validation
        validation_report['validations']['database_schema'] = self._validate_database_schema()

        # 2. API Collection Validation
        validation_report['validations']['api_collection'] = self._validate_api_collection(date)

        # 3. Data Pipeline Consistency Validation
        validation_report['validations']['pipeline_consistency'] = self._validate_pipeline_consistency(date)

        # 4. Field Completeness Validation
        validation_report['validations']['field_completeness'] = self._validate_field_completeness(date)

        # 5. Feature Quality Validation
        validation_report['validations']['feature_quality'] = self._validate_feature_quality(date)

        # 6. Training-Prediction Consistency Validation
        validation_report['validations']['training_prediction_consistency'] = self._validate_training_consistency()

        # Determine overall status
        failed_validations = [
            name for name, result in validation_report['validations'].items()
            if result.get('status') == 'fail'
        ]

        if failed_validations:
            validation_report['overall_status'] = 'fail'
            validation_report['failed_validations'] = failed_validations
        else:
            validation_report['overall_status'] = 'pass'

        if self.verbose:
            status_emoji = "✅" if validation_report['overall_status'] == 'pass' else "❌"
            self.logger.info(f"{status_emoji} Validation completed: {validation_report['overall_status']}")

        return validation_report

    def _validate_database_schema(self) -> Dict:
        """Validate that the database schema has all required enhanced fields."""
        try:
            conn = sqlite3.connect(self.race_fetcher.db_path)
            cursor = conn.cursor()

            # Get table schema
            cursor.execute("PRAGMA table_info(daily_race)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            conn.close()

            # Required enhanced fields
            required_fields = {
                'cheque': 'REAL',
                'handi_raw': 'VARCHAR(255)',
                'reclam': 'VARCHAR(50)',
                'sex': 'VARCHAR(50)',
                'tempscourse': 'VARCHAR(100)',
                'handicap_level_score': 'REAL',
                'participants': 'JSON'
            }

            missing_fields = []
            for field, expected_type in required_fields.items():
                if field not in columns:
                    missing_fields.append(field)

            if missing_fields:
                return {
                    'status': 'fail',
                    'message': f"Missing required fields: {missing_fields}",
                    'missing_fields': missing_fields
                }

            return {
                'status': 'pass',
                'message': "Database schema validation passed",
                'total_columns': len(columns),
                'enhanced_fields_present': len(required_fields)
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"Database schema validation error: {str(e)}"
            }

    def _validate_api_collection(self, date: str) -> Dict:
        """Validate the enhanced API collection system."""
        try:
            # Test enhanced API manager
            api_manager = EnhancedAPIManager(verbose=False)

            # Perform a test collection
            collection_result = api_manager.enhanced_race_collection(date)

            if collection_result.get('status') != 'success':
                return {
                    'status': 'fail',
                    'message': f"Enhanced API collection failed: {collection_result.get('error')}",
                    'collection_result': collection_result
                }

            # Validate collection quality
            enhanced_data = collection_result.get('enhanced_data', {})
            api_calls = collection_result.get('api_calls', 0)
            total_races = len(enhanced_data)

            if total_races == 0:
                return {
                    'status': 'fail',
                    'message': "No races collected from API",
                    'api_calls': api_calls
                }

            # Check supplementary data collection
            races_with_supplementary = sum(
                1 for race_data in enhanced_data.values()
                if race_data.get('supplementary_data', {})
            )

            collection_stats = api_manager.get_collection_statistics(collection_result)

            return {
                'status': 'pass',
                'message': "Enhanced API collection validation passed",
                'total_races': total_races,
                'api_calls_used': api_calls,
                'races_with_supplementary_data': races_with_supplementary,
                'collection_efficiency': collection_stats.get('api_efficiency', 0),
                'collection_time': collection_result.get('collection_time_seconds', 0)
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"API collection validation error: {str(e)}"
            }

    def _validate_pipeline_consistency(self, date: str) -> Dict:
        """Validate that the unified pipeline produces consistent results."""
        try:
            # Get races for the date
            races = self.race_fetcher.get_races_by_date(date)

            if not races:
                return {
                    'status': 'skip',
                    'message': f"No races found for {date} to validate pipeline consistency"
                }

            consistency_issues = []
            total_races_validated = 0

            for race in races[:3]:  # Validate first 3 races to avoid excessive processing
                if not race.get('has_processed_data'):
                    continue

                # Get race participants
                race_details = self.race_fetcher.get_race_by_comp(race['comp'])
                if not race_details or not race_details.get('participants'):
                    continue

                participants = race_details['participants']
                if isinstance(participants, str):
                    participants = json.loads(participants)

                # Process through unified pipeline
                df = pd.DataFrame(participants)
                processed_df = self.unified_pipeline.process_race_data(
                    df,
                    race_context={'dist': race.get('dist', 0), 'partant': race.get('partant', 0)},
                    source="validation"
                )

                # Validate processing results
                validation_result = self.unified_pipeline._validate_feature_consistency(
                    processed_df, "validation"
                )

                if not validation_result['valid']:
                    consistency_issues.append({
                        'race_comp': race['comp'],
                        'issues': validation_result['issues']
                    })

                total_races_validated += 1

            if consistency_issues:
                return {
                    'status': 'fail',
                    'message': f"Pipeline consistency issues found in {len(consistency_issues)} races",
                    'races_validated': total_races_validated,
                    'consistency_issues': consistency_issues
                }

            return {
                'status': 'pass',
                'message': "Pipeline consistency validation passed",
                'races_validated': total_races_validated
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"Pipeline consistency validation error: {str(e)}"
            }

    def _validate_field_completeness(self, date: str) -> Dict:
        """Validate completeness of enhanced fields in stored data."""
        try:
            # Get races for the date
            races = self.race_fetcher.get_races_by_date(date)

            if not races:
                return {
                    'status': 'skip',
                    'message': f"No races found for {date} to validate field completeness"
                }

            # Enhanced field completeness tracking
            enhanced_fields_check = {
                'cheque': 0,
                'handi_raw': 0,
                'reclam': 0,
                'sex': 0,
                'tempscourse': 0,
                'handicap_level_score': 0
            }

            participant_fields_check = {
                'derniereplace': 0,
                'dernierecote': 0,
                'dernierealloc': 0,
                'txreclam': 0,
                'recordG': 0,
                'tempstot': 0,
                'ecar': 0
            }

            total_races = len(races)
            total_participants = 0

            for race in races:
                # Check race-level enhanced fields
                race_details = self.race_fetcher.get_race_by_comp(race['comp'])
                if race_details:
                    for field in enhanced_fields_check:
                        if race_details.get(field) is not None:
                            enhanced_fields_check[field] += 1

                    # Check participant-level enhanced fields
                    participants = race_details.get('participants')
                    if participants:
                        if isinstance(participants, str):
                            participants = json.loads(participants)

                        for participant in participants:
                            total_participants += 1
                            for field in participant_fields_check:
                                if participant.get(field) is not None:
                                    participant_fields_check[field] += 1

            # Calculate completeness percentages
            race_completeness = {
                field: (count / total_races) * 100
                for field, count in enhanced_fields_check.items()
            }

            participant_completeness = {
                field: (count / max(total_participants, 1)) * 100
                for field, count in participant_fields_check.items()
            }

            # Check for critical missing fields
            critical_missing = []
            if race_completeness['cheque'] < 80:  # Less than 80% have purse data
                critical_missing.append('cheque (race purse)')
            if participant_completeness['derniereplace'] < 60:  # Less than 60% have last position
                critical_missing.append('derniereplace (last race position)')

            status = 'fail' if critical_missing else 'pass'

            return {
                'status': status,
                'message': f"Field completeness validation {'failed' if critical_missing else 'passed'}",
                'total_races': total_races,
                'total_participants': total_participants,
                'race_field_completeness': race_completeness,
                'participant_field_completeness': participant_completeness,
                'critical_missing_fields': critical_missing
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"Field completeness validation error: {str(e)}"
            }

    def _validate_feature_quality(self, date: str) -> Dict:
        """Validate quality of calculated features."""
        try:
            # Get races for the date
            races = self.race_fetcher.get_races_by_date(date)

            if not races:
                return {
                    'status': 'skip',
                    'message': f"No races found for {date} to validate feature quality"
                }

            quality_issues = []
            total_participants_checked = 0

            for race in races[:2]:  # Check first 2 races to avoid excessive processing
                race_details = self.race_fetcher.get_race_by_comp(race['comp'])
                if not race_details or not race_details.get('participants'):
                    continue

                participants = race_details['participants']
                if isinstance(participants, str):
                    participants = json.loads(participants)

                df = pd.DataFrame(participants)
                total_participants_checked += len(df)

                # Check for feature quality issues
                for _, participant in df.iterrows():
                    # Check for impossible values
                    if participant.get('age', 0) > 30 or participant.get('age', 0) < 1:
                        quality_issues.append(f"Invalid age: {participant.get('age')} for horse {participant.get('cheval')}")

                    if participant.get('ratio_victoires', 0) > 1:
                        quality_issues.append(f"Invalid win ratio: {participant.get('ratio_victoires')} for horse {participant.get('cheval')}")

                    if participant.get('cotedirect', 0) < 0:
                        quality_issues.append(f"Invalid odds: {participant.get('cotedirect')} for horse {participant.get('cheval')}")

            # Limit quality issues reported
            if len(quality_issues) > 10:
                quality_issues = quality_issues[:10] + [f"... and {len(quality_issues) - 10} more issues"]

            status = 'fail' if quality_issues else 'pass'

            return {
                'status': status,
                'message': f"Feature quality validation {'failed' if quality_issues else 'passed'}",
                'total_participants_checked': total_participants_checked,
                'quality_issues_found': len(quality_issues),
                'quality_issues': quality_issues
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"Feature quality validation error: {str(e)}"
            }

    def _validate_training_consistency(self) -> Dict:
        """Validate consistency between training and daily sync pipelines."""
        try:
            # This is a conceptual validation - in practice you would compare
            # against a known training dataset schema
            expected_features = {
                'core_features': [
                    'idche', 'numero', 'cheval', 'age', 'cotedirect',
                    'victoirescheval', 'placescheval', 'coursescheval'
                ],
                'calculated_features': [
                    'ratio_victoires', 'ratio_places', 'gains_par_course',
                    'efficacite_couple', 'regularite_couple'
                ],
                'enhanced_features': [
                    'derniereplace', 'dernierecote', 'txreclam', 'recordG'
                ],
                'equipment_features': [
                    'blinkers_first_time', 'has_blinkers', 'major_shoeing_change'
                ]
            }

            # Get a sample of recent data to check feature presence
            recent_races = self.race_fetcher.get_all_daily_races()
            if not recent_races:
                return {
                    'status': 'skip',
                    'message': "No recent races to validate training consistency"
                }

            # Check first race with processed data
            sample_race = None
            for race in recent_races:
                if race.get('has_processed_data'):
                    sample_race = self.race_fetcher.get_race_by_comp(race['comp'])
                    break

            if not sample_race or not sample_race.get('participants'):
                return {
                    'status': 'skip',
                    'message': "No processed race data available for training consistency check"
                }

            participants = sample_race['participants']
            if isinstance(participants, str):
                participants = json.loads(participants)

            sample_df = pd.DataFrame(participants)
            available_features = set(sample_df.columns)

            missing_features = {}
            for category, features in expected_features.items():
                missing_in_category = [f for f in features if f not in available_features]
                if missing_in_category:
                    missing_features[category] = missing_in_category

            status = 'fail' if missing_features else 'pass'

            return {
                'status': status,
                'message': f"Training consistency validation {'failed' if missing_features else 'passed'}",
                'total_features_found': len(available_features),
                'missing_features_by_category': missing_features,
                'sample_race_comp': sample_race['comp']
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"Training consistency validation error: {str(e)}"
            }

    def generate_validation_report(self, validation_result: Dict) -> str:
        """
        Generate a human-readable validation report.

        Args:
            validation_result: Result from validate_enhanced_sync

        Returns:
            Formatted validation report string
        """
        report = []
        report.append("=" * 80)
        report.append("ENHANCED DAILY SYNC VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Date: {validation_result['date']}")
        report.append(f"Timestamp: {validation_result['timestamp']}")
        report.append(f"Overall Status: {validation_result['overall_status'].upper()}")
        report.append("")

        for validation_name, validation_data in validation_result['validations'].items():
            status_emoji = "✅" if validation_data['status'] == 'pass' else "❌" if validation_data['status'] == 'fail' else "⚠️"
            report.append(f"{status_emoji} {validation_name.replace('_', ' ').title()}")
            report.append(f"   Status: {validation_data['status']}")
            report.append(f"   Message: {validation_data['message']}")

            # Add specific details for each validation type
            if validation_name == 'api_collection' and validation_data['status'] == 'pass':
                report.append(f"   Total Races: {validation_data.get('total_races', 0)}")
                report.append(f"   API Calls: {validation_data.get('api_calls_used', 0)}")
                report.append(f"   Collection Time: {validation_data.get('collection_time', 0):.2f}s")

            elif validation_name == 'field_completeness' and validation_data['status'] in ['pass', 'fail']:
                report.append(f"   Total Races: {validation_data.get('total_races', 0)}")
                report.append(f"   Total Participants: {validation_data.get('total_participants', 0)}")
                if validation_data.get('critical_missing_fields'):
                    report.append(f"   Critical Missing: {', '.join(validation_data['critical_missing_fields'])}")

            elif validation_name == 'feature_quality' and validation_data['status'] in ['pass', 'fail']:
                report.append(f"   Participants Checked: {validation_data.get('total_participants_checked', 0)}")
                if validation_data.get('quality_issues_found', 0) > 0:
                    report.append(f"   Quality Issues Found: {validation_data['quality_issues_found']}")

            report.append("")

        if validation_result['overall_status'] == 'fail':
            report.append("FAILED VALIDATIONS:")
            for failed in validation_result.get('failed_validations', []):
                report.append(f"  - {failed.replace('_', ' ').title()}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def quick_health_check(self) -> Dict:
        """
        Perform a quick health check of the enhanced sync system.

        Returns:
            Health check result dictionary
        """
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'checks': {}
        }

        # Check database connectivity
        try:
            conn = sqlite3.connect(self.race_fetcher.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM daily_race")
            race_count = cursor.fetchone()[0]
            conn.close()

            health_status['checks']['database'] = {
                'status': 'pass',
                'total_races': race_count
            }
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['checks']['database'] = {
                'status': 'fail',
                'error': str(e)
            }

        # Check API manager
        try:
            api_manager = EnhancedAPIManager(verbose=False)
            health_status['checks']['api_manager'] = {
                'status': 'pass',
                'initialized': True
            }
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['checks']['api_manager'] = {
                'status': 'fail',
                'error': str(e)
            }

        # Check unified pipeline
        try:
            pipeline = UnifiedDataPipeline(verbose=False)
            health_status['checks']['unified_pipeline'] = {
                'status': 'pass',
                'initialized': True
            }
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['checks']['unified_pipeline'] = {
                'status': 'fail',
                'error': str(e)
            }

        return health_status