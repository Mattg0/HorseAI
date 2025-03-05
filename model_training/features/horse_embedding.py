import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import json


class HorseEmbedding:
    """
    Generates embeddings for horses based on their performance features.
    Supports both DataFrame and JSON input formats.
    """

    def __init__(self, embedding_dim=16):
        """
        Initialize the embedding generator.

        Args:
            embedding_dim: Dimension of the output embedding vector
        """
        self.embedding_dim = embedding_dim
        self.scaler = StandardScaler()
        self.feature_columns = [
            # Basic info
            'age',
            # Performance statistics
            'victoirescheval', 'placescheval', 'coursescheval',
            'pourcVictChevalHippo', 'pourcPlaceChevalHippo',
            # Derived ratios
            'ratio_victoires', 'ratio_places', 'perf_cheval_hippo',
            # Global performance metrics from musique
            'che_global_avg_pos', 'che_global_recent_perf', 'che_global_consistency',
            'che_global_trend', 'che_global_pct_top3', 'che_global_nb_courses',
            'che_global_dnf_rate',
            # Weighted performance metrics
            'che_weighted_avg_pos', 'che_weighted_recent_perf', 'che_weighted_consistency',
            'che_weighted_pct_top3', 'che_weighted_dnf_rate'
        ]

        # Simple embedding network
        self.embedding_network = nn.Sequential(
            nn.Linear(len(self.feature_columns), 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def _json_to_dataframe(self, json_data: Union[str, List[Dict]]) -> pd.DataFrame:
        """
        Convert JSON data to pandas DataFrame.

        Args:
            json_data: JSON string or list of dictionaries

        Returns:
            Pandas DataFrame
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        return pd.DataFrame(data)

    def _preprocess_features(self, data: Union[pd.DataFrame, List[Dict], str]) -> pd.DataFrame:
        """
        Preprocess horse features, handling missing values and scaling.

        Args:
            data: DataFrame, JSON string, or list of dictionaries containing participant data

        Returns:
            Preprocessed DataFrame with selected features
        """
        # Convert to DataFrame if input is JSON or list of dicts
        if not isinstance(data, pd.DataFrame):
            df = self._json_to_dataframe(data)
        else:
            df = data.copy()

        # Select relevant features
        # Handle missing columns by adding them with default values
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        features_df = df[self.feature_columns].copy()

        # Handle missing age values
        if features_df['age'].isnull().any():
            # Group by horse ID and forward fill age
            features_df['age'] = df.groupby('idche')['age'].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(5)
            )

        # Fill remaining missing values
        features_df = features_df.fillna(0)

        # Scale numerical features
        self.scaler.fit(features_df)
        scaled_features = self.scaler.transform(features_df)

        return pd.DataFrame(scaled_features, columns=self.feature_columns)

    def generate_embeddings(self, data: Union[pd.DataFrame, List[Dict], str]) -> Dict[int, np.ndarray]:
        """
        Generate embeddings for all horses in the input data.

        Args:
            data: DataFrame, JSON string, or list of dictionaries containing participant data

        Returns:
            Dictionary mapping horse IDs to embedding vectors
        """
        # Convert to DataFrame if input is JSON or list of dicts
        if not isinstance(data, pd.DataFrame):
            df = self._json_to_dataframe(data)
        else:
            df = data.copy()

        # Ensure idche column exists
        if 'idche' not in df.columns:
            raise ValueError("Input data must contain 'idche' column")

        # Preprocess features
        preprocessed_df = self._preprocess_features(df)

        # Convert to tensor
        features_tensor = torch.tensor(preprocessed_df.values, dtype=torch.float32)

        # Generate embeddings
        with torch.no_grad():
            embeddings = self.embedding_network(features_tensor).numpy()

        # Create mapping from horse ID to embedding
        horse_ids = df['idche'].values
        embedding_dict = {int(horse_ids[i]): embeddings[i] for i in range(len(horse_ids))}

        return embedding_dict

    def get_horse_representation(self, horse_id: int,
                                 embeddings_dict: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        """
        Get the embedding representation for a specific horse.

        Args:
            horse_id: ID of the horse
            embeddings_dict: Dictionary mapping horse IDs to embeddings

        Returns:
            Embedding vector or None if horse_id not found
        """
        return embeddings_dict.get(horse_id)


class EnhancedFeatureCalculator:
    """
    Enhanced version of FeatureCalculator that incorporates horse embeddings.
    Extends the existing FeatureCalculator class.
    """

    def __init__(self, embedding_dim=16):
        """
        Initialize the enhanced feature calculator.

        Args:
            embedding_dim: Dimension of horse embeddings
        """
        self.horse_embedder = HorseEmbedding(embedding_dim=embedding_dim)

    def calculate_enhanced_features(self,
                                    data: Union[pd.DataFrame, List[Dict], str]) -> Union[pd.DataFrame, List[Dict]]:
        """
        Calculate enhanced features including horse embeddings.

        Args:
            data: DataFrame, JSON string, or list of dictionaries containing raw participant data

        Returns:
            DataFrame or list of dictionaries with additional embedding features,
            matching the input format
        """
        return_json = False

        # Convert to DataFrame if input is JSON or list of dicts
        if isinstance(data, str):
            df = pd.DataFrame(json.loads(data))
            return_json = True
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
            return_json = True
        else:
            df = data.copy()

        # First calculate standard features using original FeatureCalculator
        from core.calculators.static_feature_calculator import FeatureCalculator
        result_df = FeatureCalculator.calculate_all_features(df)

        # Generate horse embeddings
        horse_embeddings = self.horse_embedder.generate_embeddings(result_df)

        # Add embedding components as features
        for i in range(self.horse_embedder.embedding_dim):
            col_name = f'horse_emb_{i}'
            result_df[col_name] = result_df['idche'].apply(
                lambda x: horse_embeddings.get(int(x), np.zeros(self.horse_embedder.embedding_dim))[i]
            )

        # Return in the same format as input
        if return_json:
            return result_df.to_dict('records')
        return result_df


def process_json_data(json_data: Union[str, List[Dict]], embedding_dim=16) -> Dict[str, Any]:
    """
    Process JSON horse data and return embeddings and enhanced features.

    Args:
        json_data: JSON string or list of dictionaries containing horse data
        embedding_dim: Dimension of the embeddings

    Returns:
        Dictionary containing horse embeddings and enhanced features
    """
    # Initialize embedding generator
    embedder = HorseEmbedding(embedding_dim=embedding_dim)

    # Generate embeddings
    embeddings = embedder.generate_embeddings(json_data)

    # Convert embeddings to serializable format
    serializable_embeddings = {
        str(horse_id): emb.tolist() for horse_id, emb in embeddings.items()
    }

    # Calculate enhanced features
    calculator = EnhancedFeatureCalculator(embedding_dim=embedding_dim)
    enhanced_features = calculator.calculate_enhanced_features(json_data)

    return {
        "embeddings": serializable_embeddings,
        "enhanced_features": enhanced_features
    }


def main():
    """
    Example usage of the horse embedding system with JSON input.
    """
    # Sample data in JSON format
    json_data = [
        {
            "idche": 101,
            "age": 5,
            "victoirescheval": 3,
            "placescheval": 8,
            "coursescheval": 20,
            "pourcVictChevalHippo": 15.0,
            "pourcPlaceChevalHippo": 40.0,
            "ratio_victoires": 0.15,
            "ratio_places": 0.4,
            "perf_cheval_hippo": 27.5,
            "che_global_avg_pos": 4.2,
            "che_global_recent_perf": 3.0,
            "che_global_consistency": 2.3,
            "che_global_trend": 1.2,
            "che_global_pct_top3": 0.45,
            "che_global_nb_courses": 18,
            "che_global_dnf_rate": 0.10,
            "che_weighted_avg_pos": 3.8,
            "che_weighted_recent_perf": 2.5,
            "che_weighted_consistency": 2.1,
            "che_weighted_pct_top3": 0.4,
            "che_weighted_dnf_rate": 0.12
        },
        {
            "idche": 102,
            "age": 7,
            "victoirescheval": 5,
            "placescheval": 12,
            "coursescheval": 44,
            "pourcVictChevalHippo": 5.56,
            "pourcPlaceChevalHippo": 16.67,
            "ratio_victoires": 0.11,
            "ratio_places": 0.27,
            "perf_cheval_hippo": 11.11,
            "che_global_avg_pos": 6.45,
            "che_global_recent_perf": 2.0,
            "che_global_consistency": 5.61,
            "che_global_trend": -2.51,
            "che_global_pct_top3": 0.32,
            "che_global_nb_courses": 38,
            "che_global_dnf_rate": 0.16,
            "che_weighted_avg_pos": 1.93,
            "che_weighted_recent_perf": 0.6,
            "che_weighted_consistency": 1.68,
            "che_weighted_pct_top3": 0.09,
            "che_weighted_dnf_rate": 0.16
        }
    ]

    # Process JSON data
    result = process_json_data(json_data, embedding_dim=8)

    print("Horse Embeddings:")
    for horse_id, emb in result["embeddings"].items():
        print(f"Horse ID {horse_id}: {emb}")

    print("\nEnhanced Features:")
    for horse in result["enhanced_features"]:
        print(f"Horse ID {horse['idche']}: Embedding components:", end=" ")
        for i in range(8):
            print(f"{horse[f'horse_emb_{i}']:.4f}", end=" ")
        print()


if __name__ == "__main__":
    main()