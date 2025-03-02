import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Union, Optional
import pickle


class JockeyEmbedding:
    """
    Generates embeddings for jockeys based on their performance statistics.
    Can be used both for training and prediction.
    """

    def __init__(self, embedding_dim: int = 8, use_pca: bool = True):
        """
        Initialize the jockey embedding generator.

        Args:
            embedding_size: Size of the embedding vector
            use_pca: Whether to use PCA for dimensionality reduction
        """
        self.embedding_dim = embedding_dim
        self.use_pca = use_pca
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=embedding_dim) if use_pca else None
        self.jockey_embeddings = {}  # Cache for jockey embeddings
        self.is_fitted = False

    def _extract_jockey_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract jockey-specific features from the dataframe.

        Args:
            df: DataFrame containing jockey data

        Returns:
            DataFrame with jockey features
        """
        # Select only jockey-relevant columns
        jockey_columns = [
            'idJockey',
            'pourcVictJockHippo', 'pourcPlaceJockHippo',
            'joc_global_avg_pos', 'joc_global_recent_perf',
            'joc_global_consistency', 'joc_global_trend',
            'joc_global_pct_top3', 'joc_global_nb_courses',
            'joc_global_dnf_rate',
            'joc_weighted_avg_pos', 'joc_weighted_recent_perf',
            'joc_weighted_consistency', 'joc_weighted_pct_top3'
        ]

        # Create a copy with only relevant columns that exist in the dataframe
        jockey_features = df[[col for col in jockey_columns if col in df.columns]].copy()

        # Handle missing columns by adding zeros
        for col in jockey_columns:
            if col not in jockey_features.columns and col != 'idJockey':
                jockey_features[col] = 0.0

        return jockey_features

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the embedding generator using all jockey data.

        Args:
            df: DataFrame containing jockey data
        """
        jockey_features = self._extract_jockey_features(df)

        # Group by jockey ID and aggregate statistics
        jockey_stats = jockey_features.groupby('idJockey').agg({
            'pourcVictJockHippo': 'mean',
            'pourcPlaceJockHippo': 'mean',
            'joc_global_avg_pos': 'mean',
            'joc_global_recent_perf': 'mean',
            'joc_global_consistency': 'mean',
            'joc_global_trend': 'mean',
            'joc_global_pct_top3': 'mean',
            'joc_global_nb_courses': 'max',
            'joc_global_dnf_rate': 'mean',
            'joc_weighted_avg_pos': 'mean',
            'joc_weighted_recent_perf': 'mean',
            'joc_weighted_consistency': 'mean',
            'joc_weighted_pct_top3': 'mean'
        }).reset_index()

        # Create additional features
        jockey_stats['experience_factor'] = np.log1p(jockey_stats['joc_global_nb_courses'])
        jockey_stats['reliability'] = 1.0 - jockey_stats['joc_global_dnf_rate']
        jockey_stats['performance_index'] = (
                                                    jockey_stats['pourcVictJockHippo'] * 3 +
                                                    jockey_stats['pourcPlaceJockHippo'] * 1.5
                                            ) / 4.5

        # Prepare data for scaling and PCA
        feature_cols = [col for col in jockey_stats.columns if col != 'idJockey']
        jockey_data = jockey_stats[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        # Scale the data
        self.scaler.fit(jockey_data)
        scaled_data = self.scaler.transform(jockey_data)

        # Apply PCA if requested
        if self.use_pca:
            self.pca.fit(scaled_data)
            transformed_data = self.pca.transform(scaled_data)
        else:
            # If not using PCA, we'll just take the first embedding_size features
            transformed_data = scaled_data[:, :self.embedding_size]

        # Create embeddings dictionary
        for i, jockey_id in enumerate(jockey_stats['idJockey']):
            self.jockey_embeddings[jockey_id] = transformed_data[i]

        self.is_fitted = True

        # Create a default embedding for unknown jockeys (mean of all jockeys)
        self.default_embedding = np.mean(transformed_data, axis=0)

    def transform(self, jockey_id: int) -> np.ndarray:
        """
        Get the embedding for a specific jockey.

        Args:
            jockey_id: ID of the jockey

        Returns:
            Embedding vector for the jockey
        """
        if not self.is_fitted:
            raise ValueError("JockeyEmbeddingGenerator must be fitted before transform")

        # Return the cached embedding if available
        if jockey_id in self.jockey_embeddings:
            return self.jockey_embeddings[jockey_id]
        else:
            # Return default embedding for unknown jockeys
            return self.default_embedding

    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add jockey embeddings to a dataframe of participants.

        Args:
            df: DataFrame containing participants with 'idJockey' column

        Returns:
            DataFrame with added jockey embedding columns
        """
        if not self.is_fitted:
            raise ValueError("JockeyEmbeddingGenerator must be fitted before transform_batch")

        result_df = df.copy()

        # Add embedding columns
        for i in range(self.embedding_size):
            col_name = f'jockey_emb_{i}'
            result_df[col_name] = 0.0

        # Populate embedding values
        for idx, row in df.iterrows():
            jockey_id = row.get('idJockey')
            if pd.notna(jockey_id):
                embedding = self.transform(int(jockey_id))
                for i in range(self.embedding_size):
                    result_df.at[idx, f'jockey_emb_{i}'] = embedding[i]

        return result_df

    def save(self, filepath: str) -> None:
        """
        Save the fitted embedding generator to a file.

        Args:
            filepath: Path where to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embedding_size': self.embedding_size,
                'use_pca': self.use_pca,
                'scaler': self.scaler,
                'pca': self.pca,
                'jockey_embeddings': self.jockey_embeddings,
                'default_embedding': self.default_embedding,
                'is_fitted': self.is_fitted
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'JockeyEmbeddingGenerator':
        """
        Load a fitted embedding generator from a file.

        Args:
            filepath: Path from where to load the model

        Returns:
            Loaded JockeyEmbeddingGenerator
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        generator = cls(
            embedding_size=data['embedding_size'],
            use_pca=data['use_pca']
        )
        generator.scaler = data['scaler']
        generator.pca = data['pca']
        generator.jockey_embeddings = data['jockey_embeddings']
        generator.default_embedding = data['default_embedding']
        generator.is_fitted = data['is_fitted']

        return generator