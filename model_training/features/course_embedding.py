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
        self.hippo_mapping = {}
        self.track_type_mapping = {}
        self.meteo_mapping = {}
        self.race_type_mapping = {}

        # Numerical features normalization parameters
        self.dist_mean = 0
        self.dist_std = 1
        self.temp_mean = 0
        self.temp_std = 1
        self.wind_mean = 0
        self.wind_std = 1

    def fit(self, course_data: pd.DataFrame):
        """
        Fit the embedding model to the course data.

        Args:
            course_data: DataFrame containing course information
        """
        # Create mappings for categorical variables
        self.hippo_mapping = {hippo: idx for idx, hippo in enumerate(course_data['hippo'].unique())}
        self.track_type_mapping = {track: idx for idx, track in enumerate(course_data['natpis'].unique())}
        self.meteo_mapping = {meteo: idx for idx, meteo in enumerate(course_data['meteo'].unique())}
        self.race_type_mapping = {race_type: idx for idx, race_type in enumerate(course_data['typec'].unique())}

        # Calculate normalization parameters for numerical features
        self.dist_mean = course_data['dist'].mean()
        self.dist_std = course_data['dist'].std() if course_data['dist'].std() > 0 else 1

        # Handle temperature, ensuring it's numeric first
        numeric_temp = pd.to_numeric(course_data['temperature'], errors='coerce')
        self.temp_mean = numeric_temp.mean()
        self.temp_std = numeric_temp.std() if numeric_temp.std() > 0 else 1

        # Handle wind force, ensuring it's numeric first
        numeric_wind = pd.to_numeric(course_data['forceVent'], errors='coerce')
        self.wind_mean = numeric_wind.mean()
        self.wind_std = numeric_wind.std() if numeric_wind.std() > 0 else 1

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

        # Extract course features
        hippo = course.get('hippo', 'N/A')
        track_type = course.get('natpis', 'N/A')
        dist = course.get('dist', self.dist_mean)
        meteo = course.get('meteo', 'N/A')
        temp = course.get('temperature', self.temp_mean)
        wind_force = course.get('forceVent', self.wind_mean)
        race_type = course.get('typec', 'N/A')

        # Convert distance to numeric if it's not
        try:
            dist = float(dist)
        except (ValueError, TypeError):
            dist = self.dist_mean

        # Convert temperature to numeric if it's not
        try:
            temp = float(temp)
        except (ValueError, TypeError):
            temp = self.temp_mean

        # Convert wind force to numeric if it's not
        try:
            wind_force = float(wind_force)
        except (ValueError, TypeError):
            wind_force = self.wind_mean

        # Normalize numerical features
        normalized_dist = (dist - self.dist_mean) / self.dist_std
        normalized_temp = (temp - self.temp_mean) / self.temp_std
        normalized_wind = (wind_force - self.wind_mean) / self.wind_std

        # Set embedding values - using a simple but effective approach
        # First section: track characteristics (hippo, track type)
        embedding[0] = self.hippo_mapping.get(hippo, len(self.hippo_mapping)) / (len(self.hippo_mapping) + 1)
        embedding[1] = self.track_type_mapping.get(track_type, len(self.track_type_mapping)) / (
                    len(self.track_type_mapping) + 1)
        embedding[2] = normalized_dist

        # Second section: weather conditions
        embedding[3] = self.meteo_mapping.get(meteo, len(self.meteo_mapping)) / (len(self.meteo_mapping) + 1)
        embedding[4] = normalized_temp
        embedding[5] = normalized_wind

        # Third section: race type
        embedding[6] = self.race_type_mapping.get(race_type, len(self.race_type_mapping)) / (
                    len(self.race_type_mapping) + 1)

        # Additional values can be derived from combinations of the above
        # For example, interaction between distance and track type
        embedding[7] = embedding[1] * embedding[2]  # Track type × distance interaction

        # Interaction between weather and track
        embedding[8] = embedding[3] * embedding[1]  # Weather × track type interaction

        # Complex interactions
        embedding[9] = embedding[2] * embedding[4] * embedding[6]  # Dist × temp × race type

        # More embedding dimensions can be filled with other features and interactions
        # The remaining dimensions can be left as zeros or filled with more complex interactions

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