import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from typing import Dict, List, Tuple, Union, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoupleEmbeddingModel(nn.Module):
    """Neural network model for learning couple embeddings."""

    def __init__(self, num_couples: int, embedding_dim: int):
        """
        Initialize the embedding model.

        Args:
            num_couples: Number of unique horse/jockey combinations
            embedding_dim: Dimension of the embedding vectors
        """
        super(CoupleEmbeddingModel, self).__init__()
        self.couple_embeddings = nn.Embedding(num_couples, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, couple_ids):
        """Forward pass through the model."""
        embeddings = self.couple_embeddings(couple_ids)
        x = self.relu(self.fc1(embeddings))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x, embeddings


class CoupleDataset(Dataset):
    """Dataset for couple embeddings training."""

    def __init__(self, couple_ids, targets):
        self.couple_ids = couple_ids
        self.targets = targets

    def __len__(self):
        return len(self.couple_ids)

    def __getitem__(self, idx):
        return self.couple_ids[idx], self.targets[idx]


class CoupleEmbedding:
    """
    Class to manage horse-jockey couple embeddings.

    This class handles:
    1. Creating unique IDs for horse/jockey combinations
    2. Training embeddings based on historical performance
    3. Retrieving embeddings for inference
    4. Saving/loading the embedding model and mappings
    """

    def __init__(self, embedding_dim: int = 32, learning_rate: float = 0.001,
                 batch_size: int = 64, epochs: int = 10):
        """
        Initialize the CoupleEmbedding class.

        Args:
            embedding_dim: Dimension of the embedding vectors
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Mappings
        self.couple_to_id = {}  # Maps (horse_id, jockey_id) to unique couple_id
        self.id_to_couple = {}  # Maps unique couple_id to (horse_id, jockey_id)
        self.next_id = 0  # Counter for generating new IDs

        # Model will be initialized during training
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_couple_id(self, horse_id: int, jockey_id: int) -> int:
        """
        Create or retrieve a unique ID for a horse/jockey combination.

        Args:
            horse_id: The horse identifier
            jockey_id: The jockey identifier

        Returns:
            Unique integer ID for this horse/jockey combination
        """
        couple = (horse_id, jockey_id)

        if couple not in self.couple_to_id:
            self.couple_to_id[couple] = self.next_id
            self.id_to_couple[self.next_id] = couple
            self.next_id += 1

        return self.couple_to_id[couple]

    def get_couple_from_id(self, couple_id: int) -> Tuple[int, int]:
        """
        Get the horse and jockey IDs from a couple_id.

        Args:
            couple_id: Unique ID for a horse/jockey combination

        Returns:
            Tuple of (horse_id, jockey_id)
        """
        if couple_id not in self.id_to_couple:
            raise KeyError(f"Couple ID {couple_id} not found in mapping")

        return self.id_to_couple[couple_id]

    def create_couple_ids_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create couple IDs for all horse/jockey combinations in a dataframe.

        Args:
            df: DataFrame containing 'idche' and 'idJockey' columns

        Returns:
            DataFrame with an additional 'couple_id' column
        """
        df = df.copy()
        df['couple_id'] = df.apply(
            lambda row: self.create_couple_id(row['idche'], row['idJockey']),
            axis=1
        )
        return df

    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'cl') -> Tuple:
        """
        Prepare data for training the embedding model.

        Args:
            df: DataFrame with race data
            target_col: Name of the column to use as target variable

        Returns:
            Tuple of (couple_ids, targets) ready for training
        """
        # Add couple_ids if not already present
        if 'couple_id' not in df.columns:
            df = self.create_couple_ids_from_df(df)

        # Convert finishing position to a binary target (1 for top 3, 0 otherwise)
        # Assuming 'cl' is the finishing position
        if target_col == 'cl':
            df['target'] = df[target_col].apply(
                lambda x: 1.0 if pd.notnull(x) and x <= 3 else 0.0
            )
        else:
            df['target'] = df[target_col]

        couple_ids = torch.tensor(df['couple_id'].values, dtype=torch.long)
        targets = torch.tensor(df['target'].values, dtype=torch.float)

        return couple_ids, targets

    def train(self, df: pd.DataFrame, target_col: str = 'cl',
              validation_split: float = 0.2) -> Dict:
        """
        Train the embedding model on historical data.

        Args:
            df: DataFrame with race data
            target_col: Name of the column to use as target variable
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with training history
        """
        # Prepare data
        couple_ids, targets = self.prepare_training_data(df, target_col)

        # Initialize model
        self.model = CoupleEmbeddingModel(
            num_couples=self.next_id,
            embedding_dim=self.embedding_dim
        ).to(self.device)

        # Split data into train and validation sets
        num_samples = len(couple_ids)
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        split = int(np.floor(validation_split * num_samples))
        train_indices, valid_indices = indices[split:], indices[:split]

        train_couple_ids = couple_ids[train_indices]
        train_targets = targets[train_indices]
        valid_couple_ids = couple_ids[valid_indices]
        valid_targets = targets[valid_indices]

        # Create data loaders
        train_dataset = CoupleDataset(train_couple_ids, train_targets)
        valid_dataset = CoupleDataset(valid_couple_ids, valid_targets)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Training loop
        history = {'train_loss': [], 'valid_loss': []}

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_couple_ids, batch_targets in train_loader:
                batch_couple_ids = batch_couple_ids.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                predictions, _ = self.model(batch_couple_ids)
                loss = criterion(predictions.squeeze(), batch_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_couple_ids.size(0)

            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)

            # Validation
            self.model.eval()
            valid_loss = 0.0

            with torch.no_grad():
                for batch_couple_ids, batch_targets in valid_loader:
                    batch_couple_ids = batch_couple_ids.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    predictions, _ = self.model(batch_couple_ids)
                    loss = criterion(predictions.squeeze(), batch_targets)

                    valid_loss += loss.item() * batch_couple_ids.size(0)

                valid_loss /= len(valid_loader.dataset)
                history['valid_loss'].append(valid_loss)

            logger.info(f"Epoch {epoch + 1}/{self.epochs} - "
                        f"Train Loss: {train_loss:.4f} - "
                        f"Valid Loss: {valid_loss:.4f}")

        return history

    def get_embedding(self, horse_id: int, jockey_id: int) -> np.ndarray:
        """
        Get the embedding vector for a specific horse/jockey combination.

        Args:
            horse_id: The horse identifier
            jockey_id: The jockey identifier

        Returns:
            Numpy array containing the embedding vector
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        couple_id = self.create_couple_id(horse_id, jockey_id)
        couple_tensor = torch.tensor([couple_id], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            _, embedding = self.model(couple_tensor)

        return embedding.cpu().numpy()[0]

    def get_all_embeddings(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Get embeddings for all known horse/jockey combinations.

        Returns:
            Dictionary mapping (horse_id, jockey_id) to embedding vectors
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        embeddings = {}
        self.model.eval()

        with torch.no_grad():
            for couple, couple_id in self.couple_to_id.items():
                couple_tensor = torch.tensor([couple_id], dtype=torch.long).to(self.device)
                _, embedding = self.model(couple_tensor)
                embeddings[couple] = embedding.cpu().numpy()[0]

        return embeddings

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add embedding features to a DataFrame.

        Args:
            df: DataFrame containing 'idche' and 'idJockey' columns

        Returns:
            DataFrame with additional embedding columns
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Add couple_ids if not already present
        if 'couple_id' not in result_df.columns:
            result_df = self.create_couple_ids_from_df(result_df)

        # Get all unique couple_ids in the DataFrame
        unique_couple_ids = result_df['couple_id'].unique()

        # Create embeddings for all unique couple_ids
        embeddings = {}
        self.model.eval()

        with torch.no_grad():
            for couple_id in unique_couple_ids:
                couple_tensor = torch.tensor([couple_id], dtype=torch.long).to(self.device)
                _, embedding = self.model(couple_tensor)
                embeddings[couple_id] = embedding.cpu().numpy()[0]

        # Add embedding columns to DataFrame
        for i in range(self.embedding_dim):
            col_name = f'couple_emb_{i}'
            result_df[col_name] = result_df['couple_id'].map(
                lambda x: embeddings[x][i] if x in embeddings else 0.0
            )

        return result_df

    def save(self, filepath: str) -> None:
        """
        Save the embedding model and mappings to disk.

        Args:
            filepath: Path to save the model and mappings
        """
        save_dict = {
            'couple_to_id': self.couple_to_id,
            'id_to_couple': self.id_to_couple,
            'next_id': self.next_id,
            'embedding_dim': self.embedding_dim,
        }

        # Save mappings
        with open(f"{filepath}_mappings.pkl", 'wb') as f:
            pickle.dump(save_dict, f)

        # Save model if trained
        if self.model is not None:
            torch.save(self.model.state_dict(), f"{filepath}_model.pt")

        logger.info(f"Saved model and mappings to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load the embedding model and mappings from disk.

        Args:
            filepath: Path to load the model and mappings from
        """
        # Load mappings
        with open(f"{filepath}_mappings.pkl", 'rb') as f:
            load_dict = pickle.load(f)

        self.couple_to_id = load_dict['couple_to_id']
        self.id_to_couple = load_dict['id_to_couple']
        self.next_id = load_dict['next_id']
        self.embedding_dim = load_dict['embedding_dim']

        # Initialize and load model
        model_path = f"{filepath}_model.pt"
        if os.path.exists(model_path):
            self.model = CoupleEmbeddingModel(
                num_couples=self.next_id,
                embedding_dim=self.embedding_dim
            ).to(self.device)

            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            logger.info(f"Loaded model and mappings from {filepath}")
        else:
            logger.warning(f"Model file not found at {model_path}. Only mappings loaded.")