import pandas as pd
import numpy as np
import json
import sqlite3
import logging
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path

# Import feature calculators for consistency
from core.calculators.static_feature_calculator import FeatureCalculator
from core.calculators.musique_calculation import MusiqueFeatureExtractor
from core.data_cleaning.tabnet_cleaner import TabNetDataCleaner

# Import transformer for handicap encoding
try:
    from core.transformers.handicap_encoder import HandicapEncoder
except ImportError:
    HandicapEncoder = None


class UnifiedDataPipeline:
    """
    Unified data preparation pipeline ensuring consistency between training and daily sync.
    Applies identical data cleaning, feature calculation, and validation across all data sources.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the unified data pipeline.

        Args:
            verbose: Whether to output verbose logs
        """
        self.verbose = verbose

        # Setup logging
        self.logger = logging.getLogger("UnifiedDataPipeline")
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

        # Initialize components
        self.feature_calculator = FeatureCalculator()
        self.tabnet_cleaner = TabNetDataCleaner()
        self.musique_extractor = MusiqueFeatureExtractor()

        # Initialize handicap encoder if available
        self.handicap_encoder = HandicapEncoder() if HandicapEncoder else None

        # Expected feature schema for validation
        self.expected_features = self._define_expected_features()

        if self.verbose:
            self.logger.info("✅ Unified data pipeline initialized")

    def _define_expected_features(self) -> Dict[str, str]:
        """
        Define the expected feature schema for validation.

        Returns:
            Dictionary mapping feature names to expected data types
        """
        return {
            # Core participant fields
            'idche': 'int64',
            'numero': 'int64',
            'cheval': 'object',
            'age': 'float64',
            'cotedirect': 'float64',

            # Career statistics
            'victoirescheval': 'float64',
            'placescheval': 'float64',
            'coursescheval': 'float64',
            'gainsCarriere': 'float64',

            # Performance ratios (calculated)
            'ratio_victoires': 'float64',
            'ratio_places': 'float64',
            'gains_par_course': 'float64',

            # Jockey statistics
            'idJockey': 'int64',
            'pourcVictJockHippo': 'float64',
            'pourcPlaceJockHippo': 'float64',

            # Couple statistics (calculated)
            'nbCourseCouple': 'float64',
            'nbVictCouple': 'float64',
            'efficacite_couple': 'float64',
            'regularite_couple': 'float64',

            # Enhanced fields (Phase 2)
            'derniereplace': 'float64',
            'dernierecote': 'float64',
            'dernierealloc': 'float64',
            'txreclam': 'float64',
            'dernieredist': 'float64',
            'derniernbpartants': 'float64',
            'tempstot': 'float64',
            'ecar': 'float64',
            'vha': 'float64',

            # Equipment features (calculated)
            'blinkers_first_time': 'float64',
            'has_blinkers': 'float64',
            'blinkers_added': 'float64',
            'shoeing_first_time': 'float64',
            'major_shoeing_change': 'float64',

            # Musique features (calculated - will be prefixed)
            # These will be dynamically validated based on presence

            # Phase 2 competitive features (calculated)
            'competitive_index': 'float64',
            'field_strength_score': 'float64',
            'purse_impact_factor': 'float64'
        }

    def process_race_data(self,
                         race_data: Union[pd.DataFrame, List[Dict]],
                         race_context: Optional[Dict] = None,
                         source: str = "unknown") -> pd.DataFrame:
        """
        Process race data through the unified pipeline ensuring training-prediction consistency.

        Args:
            race_data: Race participant data (DataFrame or list of dicts)
            race_context: Additional race-level context information
            source: Data source identifier ("training", "daily_sync", etc.)

        Returns:
            Processed DataFrame with consistent feature set
        """
        if self.verbose:
            self.logger.info(f"🔄 Processing {source} data through unified pipeline")

        # Step 1: Convert to DataFrame if needed
        if isinstance(race_data, list):
            df = pd.DataFrame(race_data)
        else:
            df = race_data.copy()

        if len(df) == 0:
            self.logger.warning("⚠️  Empty data provided to pipeline")
            return df

        # Step 2: Apply comprehensive data cleaning (TabNet compatibility)
        df = self.tabnet_cleaner.comprehensive_data_cleaning(df, verbose=self.verbose)

        # Step 3: Calculate all static features (Phase 1)
        df = self._apply_static_feature_calculation(df)

        # Step 4: Calculate musique features
        df = self._apply_musique_features(df)

        # Step 5: Calculate Phase 2 derived features
        df = self._apply_advanced_features(df, race_context)

        # Step 6: Apply consistent missing value handling
        df = self._apply_consistent_missing_value_handling(df)

        # Step 7: Validate feature consistency
        validation_result = self._validate_feature_consistency(df, source)

        if not validation_result['valid']:
            self.logger.warning(f"⚠️  Feature consistency validation failed for {source}: {validation_result['issues']}")

        if self.verbose:
            self.logger.info(f"✅ Unified pipeline completed for {source}: {len(df)} participants, {len(df.columns)} features")

        return df

    def _apply_static_feature_calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply static feature calculations consistently."""
        try:
            processed_df = self.feature_calculator.calculate_all_features(df)
            if self.verbose:
                self.logger.info(f"📊 Applied static feature calculations: {len(processed_df.columns)} total features")
            return processed_df
        except Exception as e:
            self.logger.error(f"❌ Error in static feature calculation: {str(e)}")
            return df

    def _apply_musique_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply musique feature extraction consistently."""
        processed_df = df.copy()

        for index, row in processed_df.iterrows():
            try:
                # Horse musique features
                if 'musiqueche' in row and row['musiqueche']:
                    race_type = row.get('typec', 'Plat')
                    che_features = self.musique_extractor.extract_features(row['musiqueche'], race_type)

                    # Add horse musique features with consistent naming
                    for category, features in che_features.items():
                        if isinstance(features, dict):
                            for feature_name, value in features.items():
                                col_name = f"che_{category}_{feature_name}"
                                processed_df.at[index, col_name] = value
                        else:
                            # Handle non-dict features
                            col_name = f"che_{category}"
                            processed_df.at[index, col_name] = features

                # Jockey musique features
                jockey_musique = row.get('musiquejoc', row.get('musiqueche', ''))
                if jockey_musique:
                    race_type = row.get('typec', 'Plat')
                    joc_features = self.musique_extractor.extract_features(jockey_musique, race_type)

                    # Add jockey musique features with consistent naming
                    for category, features in joc_features.items():
                        if isinstance(features, dict):
                            for feature_name, value in features.items():
                                col_name = f"joc_{category}_{feature_name}"
                                processed_df.at[index, col_name] = value
                        else:
                            col_name = f"joc_{category}"
                            processed_df.at[index, col_name] = features

            except Exception as e:
                self.logger.warning(f"⚠️  Error processing musique for row {index}: {str(e)}")

        if self.verbose:
            musique_cols = [col for col in processed_df.columns if col.startswith(('che_', 'joc_'))]
            self.logger.info(f"🎵 Applied musique features: {len(musique_cols)} musique-derived features")

        return processed_df

    def _apply_advanced_features(self, df: pd.DataFrame, race_context: Optional[Dict]) -> pd.DataFrame:
        """Apply Phase 2 derived features consistently."""
        processed_df = df.copy()

        for index, row in processed_df.iterrows():
            try:
                participant = row.to_dict()

                # Extract race-level information for Phase 2 features
                race_info = {
                    'cheque': race_context.get('cheque', 0) if race_context else participant.get('cheque', 0),
                    'partant': race_context.get('partant', 0) if race_context else participant.get('partant', 0),
                    'dist': race_context.get('dist', 0) if race_context else participant.get('dist', 0),
                    'handicap_level_score': self._calculate_handicap_score(
                        race_context.get('handi') if race_context else participant.get('handi')
                    )
                }

                # Calculate advanced features (formerly Phase 2 features, now in FeatureCalculator)
                class_features = FeatureCalculator.calculate_class_movement_features(participant, race_info)
                speed_features = FeatureCalculator.calculate_speed_figure_features(participant, race_info)
                form_features = FeatureCalculator.calculate_form_context_features(participant, race_info)
                connection_features = FeatureCalculator.calculate_connection_features(participant)
                competition_features = FeatureCalculator.calculate_competition_context_features(participant, race_info)
                interaction_features = FeatureCalculator.calculate_interaction_features(participant, race_info)

                # Merge all advanced features
                advanced_features = {**class_features, **speed_features, **form_features,
                                   **connection_features, **competition_features, **interaction_features}
                for key, value in advanced_features.items():
                    processed_df.at[index, key] = value

            except Exception as e:
                self.logger.warning(f"⚠️  Error calculating advanced features for row {index}: {str(e)}")

        if self.verbose:
            advanced_cols = [col for col in processed_df.columns if col in [
                'class_drop_pct', 'speed_figure_proxy', 'form_momentum', 'trainer_change',
                'field_size_change', 'distance_comfort', 'recence_x_class_drop'
            ]]
            self.logger.info(f"⚙️  Applied advanced features: {len(advanced_cols)} features")

        return processed_df

    def _calculate_handicap_score(self, handi_raw: str) -> float:
        """Calculate handicap score using consistent encoding."""
        if not handi_raw:
            return 0.0

        if self.handicap_encoder:
            encoded_result = HandicapEncoder.parse_handicap_text(handi_raw)
            return float(encoded_result['handicap_level_score'])
        else:
            # Fallback simple encoding
            handi_str = str(handi_raw).upper().strip()
            simple_mapping = {
                'LISTE': 5.0, 'GROUP': 4.5, 'GROUPE': 4.5, 'GRADED': 4.0,
                'HANDICAP': 3.0, 'CLAIMING': 2.0, 'MAIDEN': 1.0, 'CONDITIONS': 3.5
            }
            for key, value in simple_mapping.items():
                if key in handi_str:
                    return value
            return 2.5

    def _apply_consistent_missing_value_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply consistent missing value handling across all data sources."""
        processed_df = df.copy()

        # Define consistent defaults (matching training pipeline)
        consistent_defaults = {
            'age': 5.0,
            'victoirescheval': 0.0,
            'placescheval': 0.0,
            'coursescheval': 0.0,
            'gainsCarriere': 0.0,
            'ratio_victoires': 0.0,
            'ratio_places': 0.0,
            'gains_par_course': 0.0,
            'efficacite_couple': 0.0,
            'regularite_couple': 0.0,
            'perf_cheval_hippo': 0.0,
            'perf_jockey_hippo': 0.0,
            'derniereplace': 10.0,
            'dernierecote': 10.0,
            'dernierealloc': 0.0,
            'txreclam': 0.0,
            'dernieredist': 2000.0,
            'derniernbpartants': 10.0,
            'tempstot': 0.0,
            'ecar': 0.0,
            'vha': 0.0
        }

        # Apply consistent defaults
        for field, default in consistent_defaults.items():
            if field in processed_df.columns:
                processed_df[field] = processed_df[field].fillna(default)

        # Handle string fields consistently
        string_fields = ['musiqueche', 'musiquejoc', 'cheval']
        for field in string_fields:
            if field in processed_df.columns:
                processed_df[field] = processed_df[field].fillna('')

        return processed_df

    def _validate_feature_consistency(self, df: pd.DataFrame, source: str) -> Dict:
        """
        Validate that the processed data maintains feature consistency.

        Args:
            df: Processed DataFrame
            source: Data source identifier

        Returns:
            Validation result dictionary
        """
        validation_result = {
            'valid': True,
            'issues': [],
            'feature_count': len(df.columns),
            'source': source
        }

        # Check for critical missing features
        critical_features = [
            'idche', 'numero', 'cheval', 'age', 'cotedirect',
            'ratio_victoires', 'ratio_places', 'efficacite_couple'
        ]

        missing_critical = [feat for feat in critical_features if feat not in df.columns]
        if missing_critical:
            validation_result['valid'] = False
            validation_result['issues'].append(f"Missing critical features: {missing_critical}")

        # Check for unexpected data types
        numeric_features = [
            'age', 'cotedirect', 'ratio_victoires', 'ratio_places',
            'victoirescheval', 'placescheval', 'coursescheval'
        ]

        for feature in numeric_features:
            if feature in df.columns:
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    validation_result['issues'].append(f"Feature {feature} is not numeric")

        # Check for excessive missing values
        for column in df.columns:
            missing_pct = df[column].isnull().sum() / len(df)
            if missing_pct > 0.8:  # More than 80% missing
                validation_result['issues'].append(f"Feature {column} has {missing_pct:.1%} missing values")

        # Validate feature ranges
        range_validations = {
            'age': (1, 30),
            'ratio_victoires': (0, 1),
            'ratio_places': (0, 1)
        }

        for feature, (min_val, max_val) in range_validations.items():
            if feature in df.columns:
                out_of_range = ((df[feature] < min_val) | (df[feature] > max_val)).sum()
                if out_of_range > 0:
                    validation_result['issues'].append(
                        f"Feature {feature} has {out_of_range} values outside expected range [{min_val}, {max_val}]"
                    )

        if validation_result['issues']:
            validation_result['valid'] = False

        return validation_result

    def compare_pipeline_outputs(self,
                                training_df: pd.DataFrame,
                                daily_df: pd.DataFrame) -> Dict:
        """
        Compare outputs from training and daily sync pipelines to ensure consistency.

        Args:
            training_df: DataFrame from training pipeline
            daily_df: DataFrame from daily sync pipeline

        Returns:
            Comparison result dictionary
        """
        comparison = {
            'consistent': True,
            'issues': [],
            'feature_comparison': {},
            'statistics': {}
        }

        # Compare feature sets
        training_features = set(training_df.columns)
        daily_features = set(daily_df.columns)

        missing_in_daily = training_features - daily_features
        extra_in_daily = daily_features - training_features

        if missing_in_daily:
            comparison['consistent'] = False
            comparison['issues'].append(f"Daily sync missing features: {list(missing_in_daily)}")

        if extra_in_daily:
            comparison['issues'].append(f"Daily sync has extra features: {list(extra_in_daily)}")

        # Compare feature statistics for common features
        common_features = training_features & daily_features
        numeric_features = [col for col in common_features if pd.api.types.is_numeric_dtype(training_df[col])]

        for feature in numeric_features:
            train_stats = training_df[feature].describe()
            daily_stats = daily_df[feature].describe()

            comparison['feature_comparison'][feature] = {
                'training_mean': train_stats['mean'],
                'daily_mean': daily_stats['mean'],
                'training_std': train_stats['std'],
                'daily_std': daily_stats['std']
            }

            # Check for significant differences
            mean_diff = abs(train_stats['mean'] - daily_stats['mean'])
            if mean_diff > train_stats['std']:  # Difference greater than 1 standard deviation
                comparison['issues'].append(
                    f"Feature {feature} has significant mean difference: "
                    f"training={train_stats['mean']:.3f}, daily={daily_stats['mean']:.3f}"
                )

        comparison['statistics'] = {
            'training_features': len(training_features),
            'daily_features': len(daily_features),
            'common_features': len(common_features),
            'training_rows': len(training_df),
            'daily_rows': len(daily_df)
        }

        return comparison

    def generate_feature_schema(self, df: pd.DataFrame) -> Dict:
        """
        Generate a feature schema from processed data for validation.

        Args:
            df: Processed DataFrame

        Returns:
            Feature schema dictionary
        """
        schema = {
            'total_features': len(df.columns),
            'feature_types': {},
            'feature_categories': {
                'core': [],
                'calculated': [],
                'musique': [],
                'equipment': [],
                'advanced': []
            },
            'missing_value_stats': {}
        }

        for column in df.columns:
            # Feature type
            schema['feature_types'][column] = str(df[column].dtype)

            # Missing value statistics
            missing_count = df[column].isnull().sum()
            schema['missing_value_stats'][column] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_count / len(df))
            }

            # Categorize features
            if column.startswith(('che_', 'joc_')):
                schema['feature_categories']['musique'].append(column)
            elif 'blinkers' in column or 'shoeing' in column or 'equipment' in column:
                schema['feature_categories']['equipment'].append(column)
            elif column in ['competitive_index', 'field_strength_score', 'purse_impact_factor']:
                schema['feature_categories']['advanced'].append(column)
            elif column in ['ratio_victoires', 'ratio_places', 'efficacite_couple', 'perf_cheval_hippo']:
                schema['feature_categories']['calculated'].append(column)
            else:
                schema['feature_categories']['core'].append(column)

        return schema