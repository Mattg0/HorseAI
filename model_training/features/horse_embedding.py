import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Set
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import json


class HorseEmbeddingGenerator:
    """
    Generates embeddings for horses based on their performance features.
    Supports both DataFrame and JSON input formats.
    Includes proprietaire (owner) encoding.
    """

    def __init__(self, embedding_dim=16, proprietaire_emb_dim=4):
        """
        Initialize the embedding generator.

        Args:
            embedding_dim: Dimension of the output embedding vector
            proprietaire_emb_dim: Dimension for proprietaire embedding
        """
        self.embedding_dim = embedding_dim
        self.proprietaire_emb_dim = proprietaire_emb_dim
        self.scaler = StandardScaler()

        # Numerical feature columns
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

        # Owner encoding
        self.proprietaire_encoding = {}
        self.next_proprietaire_id = 0

        # Simple embedding network - takes numerical features + proprietaire embedding
        self.embedding_network = nn.Sequential(
            nn.Linear(len(self.feature_columns) + proprietaire_emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

        # Simple embedding layer for proprietaire
        self.proprietaire_embedding = nn.Embedding(1000, proprietaire_emb_dim)  # Start with max 1000 owners

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

    def _encode_proprietaires(self, df: pd.DataFrame) -> None:
        """
        Create integer encodings for proprietaire strings.

        Args:
            df: DataFrame containing proprietaire data
        """
        if 'proprietaire' not in df.columns:
            return

        # Get unique proprietaires
        unique_proprietaires = set(df['proprietaire'].dropna().unique())

        # Add new proprietaires to encoding
        for prop in unique_proprietaires:
            if prop and prop not in self.proprietaire_encoding:
                self.proprietaire_encoding[prop] = self.next_proprietaire_id
                self.next_proprietaire_id += 1

                # Resize embedding layer if needed
                if self.next_proprietaire_id >= self.proprietaire_embedding.num_embeddings:
                    old_embedding = self.proprietaire_embedding
                    self.proprietaire_embedding = nn.Embedding(
                        self.next_proprietaire_id + 1000,  # Add buffer
                        self.proprietaire_emb_dim
                    )
                    if old_embedding.weight.data.shape[0] > 0:
                        # Copy existing weights
                        self.proprietaire_embedding.weight.data[:old_embedding.weight.data.shape[0]] = \
                            old_embedding.weight.data

    def _get_proprietaire_embeddings(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Get embeddings for proprietaires in the DataFrame.

        Args:
            df: DataFrame containing proprietaire column

        Returns:
            Tensor of proprietaire embeddings
        """
        if 'proprietaire' not in df.columns:
            # Return zeros if no proprietaire column
            return torch.zeros((len(df), self.proprietaire_emb_dim))

        # Map proprietaires to IDs, using -1 for unknown/missing
        prop_ids = df['proprietaire'].apply(
            lambda x: self.proprietaire_encoding.get(x, 0) if pd.notna(x) else 0
        ).values

        # Convert to tensor
        prop_ids_tensor = torch.tensor(prop_ids, dtype=torch.long)

        # Get embeddings
        with torch.no_grad():
            prop_embeddings = self.proprietaire_embedding(prop_ids_tensor)

        return prop_embeddings

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

        # Encode proprietaires
        self._encode_proprietaires(df)

        # Preprocess numerical features
        preprocessed_df = self._preprocess_features(df)

        # Get proprietaire embeddings
        proprietaire_embeddings = self._get_proprietaire_embeddings(df)

        # Combine numerical features with proprietaire embeddings
        features_tensor = torch.tensor(preprocessed_df.values, dtype=torch.float32)
        combined_features = torch.cat([features_tensor, proprietaire_embeddings], dim=1)

        # Generate embeddings
        with torch.no_grad():
            embeddings = self.embedding_network(combined_features).numpy()

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

    def get_proprietaire_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about proprietaires in the embedding space.

        Returns:
            Dictionary of proprietaire stats
        """
        stats = {}

        # Calculate stats if we have enough proprietaires
        if len(self.proprietaire_encoding) > 1:
            # Create tensor of all proprietaire IDs
            prop_ids = torch.tensor(list(self.proprietaire_encoding.values()), dtype=torch.long)

            # Get embeddings
            with torch.no_grad():
                prop_embeddings = self.proprietaire_embedding(prop_ids).numpy()

            # Calculate stats for each proprietaire
            for prop, idx in self.proprietaire_encoding.items():
                emb = prop_embeddings[prop_ids == idx].reshape(-1)
                stats[prop] = {
                    "norm": float(np.linalg.norm(emb)),
                    "mean": float(np.mean(emb)),
                    "min": float(np.min(emb)),
                    "max": float(np.max(emb))
                }

        return stats

    def save_model(self, path: str) -> None:
        """
        Save the embedding model and proprietaire encodings.

        Args:
            path: Path to save the model
        """
        model_data = {
            "embedding_network": self.embedding_network.state_dict(),
            "proprietaire_embedding": self.proprietaire_embedding.state_dict(),
            "proprietaire_encoding": self.proprietaire_encoding,
            "next_proprietaire_id": self.next_proprietaire_id,
            "feature_columns": self.feature_columns,
            "scaler_mean": self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            "scaler_var": self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') else None,
            "embedding_dim": self.embedding_dim,
            "proprietaire_emb_dim": self.proprietaire_emb_dim
        }

        torch.save(model_data, path)

    @classmethod
    def load_model(cls, path: str) -> 'HorseEmbeddingGenerator':
        """
        Load a saved embedding model.

        Args:
            path: Path to the saved model

        Returns:
            Loaded HorseEmbeddingGenerator instance
        """
        model_data = torch.load(path)

        # Create instance with saved dimensions
        instance = cls(
            embedding_dim=model_data["embedding_dim"],
            proprietaire_emb_dim=model_data["proprietaire_emb_dim"]
        )

        # Restore feature columns
        instance.feature_columns = model_data["feature_columns"]

        # Restore proprietaire data
        instance.proprietaire_encoding = model_data["proprietaire_encoding"]
        instance.next_proprietaire_id = model_data["next_proprietaire_id"]

        # Restore proprietaire embedding with correct size
        instance.proprietaire_embedding = nn.Embedding(
            max(1000, instance.next_proprietaire_id + 100),
            instance.proprietaire_emb_dim
        )
        instance.proprietaire_embedding.load_state_dict(model_data["proprietaire_embedding"])

        # Restore network architecture based on shapes
        input_dim = len(instance.feature_columns) + instance.proprietaire_emb_dim
        instance.embedding_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, instance.embedding_dim)
        )
        instance.embedding_network.load_state_dict(model_data["embedding_network"])

        # Restore scaler if available
        if model_data["scaler_mean"] is not None and model_data["scaler_var"] is not None:
            instance.scaler.mean_ = np.array(model_data["scaler_mean"])
            instance.scaler.var_ = np.array(model_data["scaler_var"])
            instance.scaler.scale_ = np.sqrt(instance.scaler.var_)

        return instance