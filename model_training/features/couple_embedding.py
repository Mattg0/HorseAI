# model_training/features/couple_embedding.py

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
                 batch_size: int = 64, epochs: int = 4):
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
        self._reserved_id_for_unknown = -1  # Special ID for missing/unknown couples

        # Model will be initialized during training
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Performance stats for couples
        self.couple_stats = {}  # Maps couple_id to performance statistics

        # Training history
        self.history = None

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self.model is not None

    def create_couple_id(self, horse_id: int, jockey_id: int, inference_mode: bool = False) -> int:
        """
        Create or retrieve a unique ID for a horse/jockey combination.
        Handles missing or invalid inputs by returning a reserved ID.

        Args:
            horse_id: The horse identifier
            jockey_id: The jockey identifier
            inference_mode: If True, don't create new IDs (for prediction)

        Returns:
            Unique integer ID for this horse/jockey combination
        """
        # Handle missing or invalid values
        if pd.isna(horse_id) or pd.isna(jockey_id):
            return self._reserved_id_for_unknown

        # Convert to integers if needed
        try:
            horse_id = int(horse_id)
            jockey_id = int(jockey_id)
        except (ValueError, TypeError):
            return self._reserved_id_for_unknown

        # Create or retrieve ID
        couple = (horse_id, jockey_id)
        if couple not in self.couple_to_id:
            if inference_mode:
                # During inference, don't create new couples - use unknown ID
                logger.debug(f"Unknown couple ({horse_id}, {jockey_id}) encountered during inference")
                return self._reserved_id_for_unknown
            else:
                # During training, create new couples
                self.couple_to_id[couple] = self.next_id
                self.id_to_couple[self.next_id] = couple
                self.next_id += 1

        return self.couple_to_id[couple]

    def get_couple_from_id(self, couple_id: int) -> Optional[Tuple[int, int]]:
        """
        Get the horse and jockey IDs from a couple_id.

        Args:
            couple_id: Unique ID for a horse/jockey combination

        Returns:
            Tuple of (horse_id, jockey_id) or None if not found
        """
        if couple_id == self._reserved_id_for_unknown:
            return None

        if couple_id not in self.id_to_couple:
            logger.warning(f"Couple ID {couple_id} not found in mapping")
            return None

        return self.id_to_couple[couple_id]

    def create_couple_ids_from_df(self, df: pd.DataFrame, inference_mode: bool = False) -> pd.DataFrame:
        """
        Create couple IDs for all horse/jockey combinations in a dataframe.

        Args:
            df: DataFrame containing 'idche' and 'idJockey' columns
            inference_mode: If True, don't create new IDs (for prediction)

        Returns:
            DataFrame with an additional 'couple_id' column
        """
        result_df = df.copy()

        # Check for required columns
        required_cols = ['idche', 'idJockey']
        for col in required_cols:
            if col not in result_df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        # Convert ID columns to numeric if they aren't already
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

        # Create couple_ids
        result_df['couple_id'] = result_df.apply(
            lambda row: self.create_couple_id(row['idche'], row['idJockey'], inference_mode=inference_mode),
            axis=1
        )

        return result_df

    def update_couple_stats(self, df: pd.DataFrame) -> None:
        """
        Update performance statistics for each couple.

        Args:
            df: DataFrame with race results including 'couple_id' and result columns
        """
        # Ensure df has couple_ids
        if 'couple_id' not in df.columns:
            df = self.create_couple_ids_from_df(df)

        # Calculate stats for each couple
        couple_groups = df.groupby('couple_id')

        for couple_id, group in couple_groups:
            races_total = len(group)

            # Calculate performance metrics
            wins = sum(1 for pos in group['cl'] if pd.notnull(pos) and pos == 1)
            places = sum(1 for pos in group['cl'] if pd.notnull(pos) and pos <= 3)

            # Store stats
            self.couple_stats[couple_id] = {
                'races_total': races_total,
                'wins': wins,
                'places': places,
                'win_rate': wins / races_total if races_total > 0 else 0,
                'place_rate': places / races_total if races_total > 0 else 0,
                'avg_position': group['cl'].mean() if pd.api.types.is_numeric_dtype(group['cl']) else None
            }

    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'cl',
                              target_type: str = 'classification') -> Tuple:
        """
        Prepare data for training the embedding model.

        Args:
            df: DataFrame with race data
            target_col: Name of the column to use as target variable
            target_type: 'classification', 'regression', or 'ranking'

        Returns:
            Tuple of (couple_ids, targets) ready for training
        """
        # Add couple_ids if not already present
        if 'couple_id' not in df.columns:
            df = self.create_couple_ids_from_df(df)

        # Drop rows with missing target values
        df_clean = df.dropna(subset=[target_col])

        # Handle different target types
        if target_type == 'classification':
            # Binary classification (1 for top 3, 0 otherwise)
            df_clean['target'] = df_clean[target_col].apply(
                lambda x: 1.0 if pd.notnull(x) and (
                        isinstance(x, (int, float)) and x <= 3 or
                        isinstance(x, str) and x.isdigit() and int(x) <= 3
                ) else 0.0
            )
        elif target_type == 'regression':
            # Regression (predict finish position)
            df_clean['target'] = pd.to_numeric(df_clean[target_col], errors='coerce')

            # Normalize target to 0-1 range for easier training
            max_pos = df_clean['target'].max()
            df_clean['target'] = 1.0 - (df_clean['target'] - 1) / max_pos
            df_clean['target'] = df_clean['target'].clip(0, 1)
        elif target_type == 'ranking':
            # For pairwise ranking, create pairs within each race
            # This is more complex and would be implemented separately
            # For now, fallback to classification
            logger.warning("Ranking target_type not fully implemented, using classification instead")
            df_clean['target'] = df_clean[target_col].apply(
                lambda x: 1.0 if pd.notnull(x) and x <= 3 else 0.0
            )
        else:
            raise ValueError(f"Unsupported target_type: {target_type}")

        couple_ids = torch.tensor(df_clean['couple_id'].values, dtype=torch.long)
        targets = torch.tensor(df_clean['target'].values, dtype=torch.float)

        return couple_ids, targets

    def train(self, df: pd.DataFrame, target_col: str = 'cl', target_type: str = 'classification',
              validation_split: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train the embedding model on historical data.

        Args:
            df: DataFrame with race data
            target_col: Name of the column to use as target variable
            target_type: 'classification', 'regression', or 'ranking'
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training history
        """
        # Set seed for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Update couple statistics
        if target_col == 'cl':
            self.update_couple_stats(df)

        # Prepare data
        couple_ids, targets = self.prepare_training_data(df, target_col, target_type)

        # Check if we have enough data
        if len(couple_ids) < 10:
            logger.warning(f"Not enough data for training (only {len(couple_ids)} samples). Need at least 10.")
            return {'train_loss': [], 'valid_loss': []}

        # Initialize model
        self.model = CoupleEmbeddingModel(
            num_couples=max(self.next_id, 10),  # Ensure we have at least 10 slots
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
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=min(self.batch_size, len(valid_dataset)),
            shuffle=False
        )

        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Training loop
        history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0

            for batch_couple_ids, batch_targets in train_loader:
                batch_couple_ids = batch_couple_ids.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                predictions, _ = self.model(batch_couple_ids)
                loss = criterion(predictions.squeeze(), batch_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_couple_ids.size(0)

                # Calculate accuracy for classification
                if target_type == 'classification':
                    pred_class = (predictions.squeeze() >= 0.5).float()
                    train_correct += (pred_class == batch_targets).sum().item()

            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)

            if target_type == 'classification':
                train_acc = train_correct / len(train_loader.dataset)
                history['train_acc'].append(train_acc)

            # Validation
            self.model.eval()
            valid_loss = 0.0
            valid_correct = 0

            with torch.no_grad():
                for batch_couple_ids, batch_targets in valid_loader:
                    batch_couple_ids = batch_couple_ids.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    predictions, _ = self.model(batch_couple_ids)
                    loss = criterion(predictions.squeeze(), batch_targets)

                    valid_loss += loss.item() * batch_couple_ids.size(0)

                    # Calculate accuracy for classification
                    if target_type == 'classification':
                        pred_class = (predictions.squeeze() >= 0.5).float()
                        valid_correct += (pred_class == batch_targets).sum().item()

                valid_loss /= len(valid_loader.dataset)
                history['valid_loss'].append(valid_loss)

                if target_type == 'classification':
                    valid_acc = valid_correct / len(valid_loader.dataset)
                    history['valid_acc'].append(valid_acc)

                    logger.info(f"Epoch {epoch + 1}/{self.epochs} - "
                                f"Train Loss: {train_loss:.4f} - "
                                f"Valid Loss: {valid_loss:.4f} - "
                                f"Train Acc: {train_acc:.4f} - "
                                f"Valid Acc: {valid_acc:.4f}")
                else:
                    logger.info(f"Epoch {epoch + 1}/{self.epochs} - "
                                f"Train Loss: {train_loss:.4f} - "
                                f"Valid Loss: {valid_loss:.4f}")

        # Store history for later analysis
        self.history = history
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
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")

        # Handle missing or invalid values
        if pd.isna(horse_id) or pd.isna(jockey_id):
            return np.zeros(self.embedding_dim)

        couple_id = self.create_couple_id(horse_id, jockey_id)
        if couple_id == self._reserved_id_for_unknown:
            return np.zeros(self.embedding_dim)

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
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")

        embeddings = {}
        self.model.eval()

        with torch.no_grad():
            for couple, couple_id in self.couple_to_id.items():
                couple_tensor = torch.tensor([couple_id], dtype=torch.long).to(self.device)
                _, embedding = self.model(couple_tensor)
                embeddings[couple] = embedding.cpu().numpy()[0]

        return embeddings

    def transform_df(self, df: pd.DataFrame, max_dim: Optional[int] = None) -> pd.DataFrame:
        """
        Add embedding features to a DataFrame.

        Args:
            df: DataFrame containing 'idche' and 'idJockey' columns
            max_dim: Maximum embedding dimensions to include (default: use all)

        Returns:
            DataFrame with additional embedding columns
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")

        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Add couple_ids if not already present (use inference mode during prediction)
        if 'couple_id' not in result_df.columns:
            result_df = self.create_couple_ids_from_df(result_df, inference_mode=True)

        # Get all unique couple_ids in the DataFrame
        unique_couple_ids = result_df['couple_id'].unique()

        # Create embeddings for all unique couple_ids
        embeddings = {}
        self.model.eval()

        with torch.no_grad():
            for couple_id in unique_couple_ids:
                if couple_id == self._reserved_id_for_unknown:
                    embeddings[couple_id] = np.zeros(self.embedding_dim)
                else:
                    # Check if couple_id is within valid range
                    if couple_id >= self.model.couple_embeddings.num_embeddings:
                        logger.warning(f"Couple ID {couple_id} exceeds model capacity "
                                     f"({self.model.couple_embeddings.num_embeddings}). Using zero embedding.")
                        embeddings[couple_id] = np.zeros(self.embedding_dim)
                    else:
                        try:
                            couple_tensor = torch.tensor([couple_id], dtype=torch.long).to(self.device)
                            _, embedding = self.model(couple_tensor)
                            embeddings[couple_id] = embedding.cpu().numpy()[0]
                        except Exception as e:
                            logger.warning(f"Error getting embedding for couple_id {couple_id}: {e}")
                            embeddings[couple_id] = np.zeros(self.embedding_dim)

        # Determine embedding dimensions to use
        dims_to_use = min(self.embedding_dim, max_dim if max_dim is not None else self.embedding_dim)

        # Add embedding columns to DataFrame
        for i in range(dims_to_use):
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
            'couple_stats': self.couple_stats,
            'history': self.history
        }

        # Save mappings
        with open(f"{filepath}_mappings.pkl", 'wb') as f:
            pickle.dump(save_dict, f)

        # Save model if trained
        if self.is_fitted:
            torch.save(self.model.state_dict(), f"{filepath}_model.pt")

        logger.info(f"Saved model and mappings to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load the embedding model and mappings from disk.

        Args:
            filepath: Path to load the model and mappings from
        """
        # Load mappings
        try:
            with open(f"{filepath}_mappings.pkl", 'rb') as f:
                load_dict = pickle.load(f)

            self.couple_to_id = load_dict['couple_to_id']
            self.id_to_couple = load_dict['id_to_couple']
            self.next_id = load_dict['next_id']
            self.embedding_dim = load_dict['embedding_dim']

            # Load optional fields if they exist
            self.couple_stats = load_dict.get('couple_stats', {})
            self.history = load_dict.get('history', None)

            # Initialize and load model
            model_path = f"{filepath}_model.pt"
            if os.path.exists(model_path):
                self.model = CoupleEmbeddingModel(
                    num_couples=max(self.next_id, 10),
                    embedding_dim=self.embedding_dim
                ).to(self.device)

                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()

                logger.info(f"Loaded model and mappings from {filepath}")
            else:
                logger.warning(f"Model file not found at {model_path}. Only mappings loaded.")

        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'idche': [101, 102, 101, 103, 102, 101],
        'idJockey': [201, 202, 201, 203, 201, 202],
        'cl': [1, 3, 2, 5, 4, 1],  # Finishing positions
    }

    df = pd.DataFrame(data)

    # Initialize and train
    embedder = CoupleEmbedding(embedding_dim=16, epochs=5)
    embedder.train(df)

    # Get embeddings
    result_df = embedder.transform_df(df)

    print("Sample couple IDs:")
    print(df['idche'].head(3), df['idJockey'].head(3))

    print("\nEmbedding columns added:")
    print([col for col in result_df.columns if 'emb' in col])

    print("\nSample embedding values:")
    print(result_df[[col for col in result_df.columns if 'emb' in col]].head(2))