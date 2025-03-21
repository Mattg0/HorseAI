import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
import json


class CourseEmbedding:
    """
    Course embedding module that creates numerical representations of race conditions.
    This allows for more effective machine learning on race characteristics.
    """

    def __init__(self, embedding_dim=16):
        """
        Initialize the course embedding module.

        Args:
            embedding_size: Size of the embedding vector for each race
        """
        self.embedding_dim = embedding_dim

        # Dictionaries to map categorical features to numerical values
        self.feature_mappings = {
            'hippo': {'UNKNOWN': 0},
            'natpis': {'UNKNOWN': 0},
            'meteo': {'UNKNOWN': 0},
            'typec': {'UNKNOWN': 0},
            'directionVent': {'UNKNOWN': 0},
            'nebulosite': {'UNKNOWN': 0},
            'corde': {'UNKNOWN': 0},
            'pistegp': {'UNKNOWN': 0}
        }

        # Numerical features normalization parameters
        self.numeric_params = {
            'dist': {'mean': 0, 'std': 1},
            'temperature': {'mean': 0, 'std': 1},
            'forceVent': {'mean': 0, 'std': 1}
        }

        # Flag to track which features were available during fitting
        self.available_features = {}

    def fit(self, course_data):
        """
        Fit the embedding model to the course data.

        Args:
            course_data: DataFrame containing course information
        """
        # Create a copy to avoid modifying the original
        course_data = course_data.copy()

        # Track which features are available
        all_features = list(self.feature_mappings.keys()) + list(self.numeric_params.keys())
        course_columns = course_data.columns
        for feature in all_features:
            self.available_features[feature] = feature in course_columns

        available_features = [f for f, avail in self.available_features.items() if avail]
        print(f"Course features available for embedding: {available_features}")

        if not available_features:
            print("WARNING: No course features available for embedding. Using default values.")
            return

        # Handle categorical features
        for feature, mapping in self.feature_mappings.items():
            if feature in course_data.columns:
                # Convert to string and handle empty/NA values
                series = course_data[feature].astype(str)
                series = series.replace('', 'UNKNOWN').replace('nan', 'UNKNOWN')

                # Create mapping
                unique_values = series.fillna('UNKNOWN').unique()
                mapping.update({val: idx + 1 for idx, val in enumerate(unique_values) if val != 'UNKNOWN'})
                print(f"Mapped {len(mapping) - 1} unique values for {feature}")

        # Handle numerical features
        for feature, params in self.numeric_params.items():
            if feature in course_data.columns:
                # Convert to numeric, handling errors
                numeric_values = pd.to_numeric(course_data[feature], errors='coerce')

                # Check if we have enough valid values
                valid_count = numeric_values.notna().sum()
                if valid_count > 10:  # Arbitrary threshold for statistical significance
                    params['mean'] = numeric_values.mean()
                    params['std'] = numeric_values.std() if numeric_values.std() > 0 else 1
                    print(f"Calculated stats for {feature}: mean={params['mean']:.2f}, std={params['std']:.2f}")
                else:
                    print(f"Not enough valid values for {feature} ({valid_count}). Using defaults.")

        print("Course embedding fitted successfully")

    def _get_normalized_value(self, feature, value):
        """Helper to safely normalize a numeric value."""
        if pd.isna(value) or value == '':
            return 0.0

        params = self.numeric_params.get(feature, {'mean': 0, 'std': 1})

        try:
            numeric_val = float(value)
            return (numeric_val - params['mean']) / params['std']
        except (ValueError, TypeError):
            return 0.0

    def _get_categorical_value(self, feature, value):
        """Helper to safely get embedding value for a categorical feature."""
        if pd.isna(value) or value == '':
            value = 'UNKNOWN'
        elif not isinstance(value, str):
            value = str(value)

        mapping = self.feature_mappings.get(feature, {'UNKNOWN': 0})
        idx = mapping.get(value, mapping.get('UNKNOWN', 0))

        # Normalize to 0-1 range
        divisor = max(len(mapping), 1)
        return idx / divisor

    def transform_course(self, course: Dict) -> np.ndarray:
        """
        Transform a single course into its embedding representation.

        Args:
            course: Dictionary containing course information

        Returns:
            Numpy array containing the course embedding
        """
        # Initialize the embedding vector
        embedding = np.zeros(self.embedding_dim)

        # Map all features to positions in the embedding
        feature_positions = {
            'hippo': 0,
            'natpis': 1,
            'dist': 2,
            'meteo': 3,
            'temperature': 4,
            'forceVent': 5,
            'typec': 6,
            'directionVent': 7,
            'nebulosite': 8,
            'corde': 9,
            'pistegp': 10
        }

        # Process categorical features
        for feature, position in feature_positions.items():
            if position >= self.embedding_dim:
                continue  # Skip if beyond embedding dimension

            if feature in self.numeric_params:
                # Numeric feature
                value = course.get(feature, self.numeric_params[feature]['mean'])
                embedding[position] = self._get_normalized_value(feature, value)
            else:
                # Categorical feature
                value = course.get(feature, 'UNKNOWN')
                embedding[position] = self._get_categorical_value(feature, value)

        # Fill remaining dimensions with interactions if space allows
        if self.embedding_dim > 11:
            # Track type × distance interaction
            embedding[11] = embedding[1] * embedding[2]

            # Weather × track type interaction
            if self.embedding_dim > 12:
                embedding[12] = embedding[3] * embedding[1]

            # Complex interactions
            if self.embedding_dim > 13:
                embedding[13] = embedding[2] * embedding[4] * embedding[6]  # Dist × temp × race type

            if self.embedding_dim > 14:
                # Track × rail position interaction
                embedding[14] = embedding[0] * embedding[9]  # Hippo × corde

            if self.embedding_dim > 15:
                # Weather complex interaction
                embedding[15] = embedding[3] * embedding[4] * embedding[7] * embedding[
                    8]  # Meteo × temp × wind direction × nebulosite

        return embedding

    def trace_embedding_preparation(self):
        import logging

        # Set up logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filename='embedding_debug.log')

        try:
            # Assuming you're using the HorseEmbedding class
            from model_training.features.horse_embedding import HorseEmbedding

            # Create a sample DataFrame with the minimum required fields
            import pandas as pd
            test_df = pd.DataFrame({
                'idche': [1, 2, 3],
                'age': [5, 6, 4],
                'natpis': ['Herbe', 'Sable', None],  # Test with a NULL value
                # Add other required fields
            })

            logging.debug(f"Test DataFrame columns: {test_df.columns.tolist()}")

            # Try to create embeddings
            embedder = HorseEmbedding(embedding_dim=16)
            logging.debug("Created HorseEmbedding instance")

            # Generate embeddings with explicit error handling
            try:
                embeddings = embedder.generate_embeddings(test_df)
                logging.debug(f"Generated embeddings: {len(embeddings)} items")
            except Exception as e:
                logging.error(f"Error generating embeddings: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())

        except Exception as e:
            logging.error(f"Overall error: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    def transform(self, course_data: pd.DataFrame) -> np.ndarray:
        """
        Transform multiple courses into their embedding representations.

        Args:
            course_data: DataFrame containing course information

        Returns:
            Numpy array containing course embeddings
        """
        embeddings = np.zeros((len(course_data), self.embedding_dim))

        for i, (_, course) in enumerate(course_data.iterrows()):
            embeddings[i] = self.transform_course(course.to_dict())

        return embeddings

    def save(self, filepath: str):
        """
        Save the embedding model to a file.

        Args:
            filepath: Path to save the model
        """
        model_data = {
            'embedding_dim': self.embedding_dim,
            'hippo_mapping': self.hippo_mapping,
            'track_type_mapping': self.track_type_mapping,
            'meteo_mapping': self.meteo_mapping,
            'race_type_mapping': self.race_type_mapping,
            'dist_mean': float(self.dist_mean),
            'dist_std': float(self.dist_std),
            'temp_mean': float(self.temp_mean),
            'temp_std': float(self.temp_std),
            'wind_mean': float(self.wind_mean),
            'wind_std': float(self.wind_std)
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str) -> 'CourseEmbedding':
        """
        Load an embedding model from a file.

        Args:
            filepath: Path to load the model from

        Returns:
            CourseEmbedding model
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        model = cls(embedding_dim=model_data['embedding_dim'])
        model.hippo_mapping = model_data['hippo_mapping']
        model.track_type_mapping = model_data['track_type_mapping']
        model.meteo_mapping = model_data['meteo_mapping']
        model.race_type_mapping = model_data['race_type_mapping']
        model.dist_mean = model_data['dist_mean']
        model.dist_std = model_data['dist_std']
        model.temp_mean = model_data['temp_mean']
        model.temp_std = model_data['temp_std']
        model.wind_mean = model_data['wind_mean']
        model.wind_std = model_data['wind_std']

        return model


class RaceRepresentation:
    """
    Combines course embeddings with participant features to create
    a complete representation of a race for prediction models.
    """

    def __init__(self, course_embedding_model: CourseEmbedding):
        """
        Initialize the race representation module.

        Args:
            course_embedding_model: Trained course embedding model
        """
        self.course_embedding = course_embedding_model


    def create_race_representation(self, course_info: Dict, participants: List[Dict]) -> Dict:
        """
        Create a complete representation of a race.

        Args:
            course_info: Dictionary containing course information
            participants: List of dictionaries containing participant information

        Returns:
            Dictionary containing the complete race representation
        """
        # Get course embedding
        course_embedding = self.course_embedding.transform_course(course_info)

        # Process each participant
        processed_participants = []
        for participant in participants:
            # Create a copy of participant features
            processed_participant = participant.copy()

            # Add course embedding to each participant
            processed_participant['course_embedding'] = course_embedding.tolist()

            # Create participant-course interaction features
            # For example, how well does this horse perform on this track type?
            if 'pourcVictChevalHippo' in participant and course_info.get('hippo') == participant.get('hippo_recent'):
                processed_participant['hippo_advantage'] = 1.0
            else:
                processed_participant['hippo_advantage'] = 0.0

            # Add more interaction features as needed

            processed_participants.append(processed_participant)

        return {
            'course_info': course_info,
            'course_embedding': course_embedding.tolist(),
            'participants': processed_participants
        }

    def batch_process_races(self, course_infos: List[Dict], participants_by_race: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Process multiple races at once.

        Args:
            course_infos: List of dictionaries containing course information
            participants_by_race: Dictionary mapping race IDs to lists of participants

        Returns:
            List of dictionaries containing race representations
        """
        race_representations = []

        for course_info in course_infos:
            race_id = course_info['id']
            participants = participants_by_race.get(race_id, [])

            race_rep = self.create_race_representation(course_info, participants)
            race_representations.append(race_rep)

        return race_representations