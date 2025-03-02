# core/orchestrators/feature_embedding_orchestrator.py

import sqlite3
import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import argparse
import os
from datetime import datetime
import hashlib
import pickle

from utils.env_setup import AppConfig
from utils.cache_manager import CacheManager
from model_training.features.course_embedding import CourseEmbedding
from model_training.features.horse_embedding import HorseEmbedding
from model_training.features.jockey_embedding import JockeyEmbedding
from model_training.features.couple_embedding import CoupleEmbedding


class FeatureEmbeddingOrchestrator:
    """
    Orchestrator for loading historical race data from SQLite,
    applying entity embeddings, and preparing data for model training.
    Uses caching to improve performance for expensive operations.
    """

    def __init__(self, sqlite_path=None, embedding_dim=None, cache_dir=None, feature_store_dir=None):
        """
        Initialize the orchestrator with embedding models and caching.

        Args:
            sqlite_path: Path to SQLite database, if None uses default from config
            embedding_dim: Dimension size for entity embeddings, if None uses default from config
            cache_dir: Directory to store cache files, if None uses default from config
            feature_store_dir: Directory to store feature stores, if None uses default from config
        """
        # Load application configuration
        self.config = AppConfig()

        # Set paths from config or arguments
        self.sqlite_path = sqlite_path or self.config.get_sqlite_dbpath()
        self.cache_dir = cache_dir or self.config.get_cache_dir()
        self.feature_store_dir = feature_store_dir or self.config.get_feature_store_dir()

        # Set embedding dimension from config or argument
        self.embedding_dim = embedding_dim or self.config.get_default_embedding_dim()

        # Initialize caching manager
        self.cache_manager = CacheManager()

        # Initialize embedding models (will be fitted later)
        self.course_embedder = CourseEmbedding(embedding_dim=4)
        self.horse_embedder = HorseEmbedding(embedding_dim=16)
        self.jockey_embedder = JockeyEmbedding(embedding_dim=8)
        self.couple_embedder = CoupleEmbedding(embedding_dim=8)  # Smaller dim for race type

        # Track whether embeddings have been fitted
        self.embeddings_fitted = False

        # Store preprocessing parameters
        self.preprocessing_params = {}

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.feature_store_dir, exist_ok=True)

        print(f"Orchestrator initialized with:")
        print(f"  - SQLite path: {self.sqlite_path}")
        print(f"  - Cache directory: {self.cache_dir}")
        print(f"  - Feature store directory: {self.feature_store_dir}")

    # The rest of the implementation remains the same, with caching paths using self.cache_dir
    # and feature store paths using self.feature_store_dir

    def _generate_cache_key(self, prefix, params):
        """
        Generate a deterministic cache key based on function parameters.

        Args:
            prefix: Prefix for the cache key
            params: Dictionary of parameters to include in the key

        Returns:
            String cache key
        """
        # Convert params to sorted string representation
        param_str = json.dumps(params, sort_keys=True)

        # Generate hash
        hash_obj = hashlib.md5(param_str.encode())
        hash_str = hash_obj.hexdigest()

        return f"{prefix}_{hash_str}"

    def load_historical_races(self, limit=None, race_filter=None, date_filter=None, include_results=True,
                              use_cache=True):
        """
        Load historical race data from SQLite with caching.

        Args:
            limit: Optional limit for number of races to load
            race_filter: Optional filter for specific race types (e.g., 'A' for Attele)
            date_filter: Optional date filter (e.g., 'jour > "2023-01-01"')
            include_results: Whether to join with race results
            use_cache: Whether to use cached results if available

        Returns:
            DataFrame with historical race data and expanded participants
        """
        # Use specific cache file for historical data if available
        historical_cache_path = self.config.get_cache_file_path('historical_data')
        if historical_cache_path and use_cache:
            cache_dir = os.path.dirname(historical_cache_path)
            cache_file = os.path.basename(historical_cache_path)
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, cache_file)

            # Check for cache parameters
            cache_params_path = os.path.join(cache_dir, f"{os.path.splitext(cache_file)[0]}_params.json")
            cache_matches = False

            if os.path.exists(cache_path) and os.path.exists(cache_params_path):
                with open(cache_params_path, 'r') as f:
                    stored_params = json.load(f)

                # Check if stored parameters match current request
                current_params = {
                    'limit': limit,
                    'race_filter': race_filter,
                    'date_filter': date_filter,
                    'include_results': include_results
                }

                if all(stored_params.get(k) == current_params.get(k) for k in current_params):
                    cache_matches = True

            if cache_matches:
                try:
                    print(f"Loading historical data from cache: {cache_path}")
                    return pd.read_parquet(cache_path)
                except Exception as e:
                    print(f"Error loading from cache: {str(e)}")

        # Generate cache key for standard caching mechanism
        cache_params = {
            'limit': limit,
            'race_filter': race_filter,
            'date_filter': date_filter,
            'include_results': include_results,
            'db_path': self.sqlite_path
        }
        cache_key = self._generate_cache_key('historical_races', cache_params)

        # Try to get from cache
        if use_cache:
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                print("Using cached historical race data...")
                return cached_data

        print("Loading historical race data from database...")
        conn = sqlite3.connect(self.sqlite_path)

        # Base query to get race data
        if include_results:
            query = """
            SELECT hr.*, rr.ordre_arrivee
            FROM historical_races hr
            LEFT JOIN race_results rr ON hr.comp = rr.comp
            """
        else:
            query = "SELECT * FROM historical_races"

        # Build WHERE clause
        where_clauses = []
        if race_filter:
            where_clauses.append(f"hr.typec = '{race_filter}'")
        if date_filter:
            where_clauses.append(date_filter)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        if limit:
            query += f" LIMIT {limit}"

        # Execute query
        cursor = conn.cursor()
        cursor.execute(query)

        # Fetch column names and data
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()

        # Create DataFrame
        df_races = pd.DataFrame(data, columns=columns)

        # Expand participants from JSON
        expanded_df = self._expand_participants(df_races)

        conn.close()

        # Save to specialized cache if available
        if historical_cache_path and use_cache:
            try:
                # Save dataframe
                expanded_df.to_parquet(cache_path)

                # Save parameters
                with open(cache_params_path, 'w') as f:
                    json.dump(cache_params, f)

                print(f"Historical data cached to {cache_path}")
            except Exception as e:
                print(f"Error caching historical data: {str(e)}")

        # Cache with the regular caching manager too
        if use_cache:
            self.cache_manager.set(cache_key, expanded_df)

        return expanded_df

    # The rest of the methods remain the same, but we'll update the save_feature_store
    # to use the configured feature_store_dir

    def save_feature_store(self, X_train, X_val, X_test, y_train, y_val, y_test, output_dir=None, prefix=''):
        """
        Save a complete feature store with datasets and metadata.

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            y_train: Training targets
            y_val: Validation targets
            y_test: Test targets
            output_dir: Directory to save files, if None uses default from config
            prefix: Optional prefix for filenames

        Returns:
            Path to the created feature store
        """
        # Use configured feature store directory if not specified
        output_dir = output_dir or self.feature_store_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamp suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Create feature store directory
        feature_store_dir = os.path.join(output_dir, f"{prefix}feature_store_{timestamp}")
        os.makedirs(feature_store_dir, exist_ok=True)

        # 1. Save datasets as parquet files (better than CSV for ML data)
        X_train.to_parquet(os.path.join(feature_store_dir, "X_train.parquet"))
        X_val.to_parquet(os.path.join(feature_store_dir, "X_val.parquet"))
        X_test.to_parquet(os.path.join(feature_store_dir, "X_test.parquet"))

        # Save targets based on their type
        if pd.api.types.is_numeric_dtype(y_train):
            # For regression tasks
            pd.DataFrame({'target': y_train}).to_parquet(os.path.join(feature_store_dir, "y_train.parquet"))
            pd.DataFrame({'target': y_val}).to_parquet(os.path.join(feature_store_dir, "y_val.parquet"))
            pd.DataFrame({'target': y_test}).to_parquet(os.path.join(feature_store_dir, "y_test.parquet"))
        else:
            # For classification tasks
            pd.DataFrame({'target': y_train}).to_parquet(os.path.join(feature_store_dir, "y_train.parquet"))
            pd.DataFrame({'target': y_val}).to_parquet(os.path.join(feature_store_dir, "y_val.parquet"))
            pd.DataFrame({'target': y_test}).to_parquet(os.path.join(feature_store_dir, "y_test.parquet"))

        # 2. Save embedding models
        embedders_path = os.path.join(feature_store_dir, "embedders.pkl")
        with open(embedders_path, 'wb') as f:
            pickle.dump({
                'horse_embedder': self.horse_embedder,
                'jockey_embedder': self.jockey_embedder,
                'track_embedder': self.track_embedder,
                'race_type_embedder': self.race_type_embedder
            }, f)

        # 3. Save preprocessing parameters and feature metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'dataset_info': {
                'train_samples': X_train.shape[0],
                'val_samples': X_val.shape[0],
                'test_samples': X_test.shape[0],
                'feature_count': X_train.shape[1],
                'feature_names': X_train.columns.tolist(),
                'categorical_features': [col for col in X_train.select_dtypes(include=['category']).columns],
                'numerical_features': [col for col in X_train.select_dtypes(include=['float64', 'int64']).columns],
                'embedding_features': [col for col in X_train.columns if 'emb_' in col]
            },
            'preprocessing': self.preprocessing_params,
            'column_dtypes': {col: str(dtype) for col, dtype in X_train.dtypes.items()}
        }

        # Save metadata as JSON
        with open(os.path.join(feature_store_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # 4. Save feature statistics
        feature_stats = {
            'mean': X_train.mean().to_dict(),
            'std': X_train.std().to_dict(),
            'min': X_train.min().to_dict(),
            'max': X_train.max().to_dict(),
            'median': X_train.median().to_dict()
        }

        with open(os.path.join(feature_store_dir, "feature_stats.json"), 'w') as f:
            json.dump(feature_stats, f, indent=2, default=str)

        # 5. Create a README file for the feature store
        with open(os.path.join(feature_store_dir, "README.md"), 'w') as f:
            f.write(f"# Feature Store {timestamp}\n\n")
            f.write("## Dataset Information\n")
            f.write(f"- Training samples: {X_train.shape[0]}\n")
            f.write(f"- Validation samples: {X_val.shape[0]}\n")
            f.write(f"- Test samples: {X_test.shape[0]}\n")
            f.write(f"- Feature count: {X_train.shape[1]}\n\n")

            f.write("## Files\n")
            f.write("- `X_train.parquet`, `X_val.parquet`, `X_test.parquet`: Feature datasets\n")
            f.write("- `y_train.parquet`, `y_val.parquet`, `y_test.parquet`: Target variables\n")
            f.write("- `embedders.pkl`: Embedding models for entities\n")
            f.write("- `metadata.json`: Dataset and preprocessing metadata\n")
            f.write("- `feature_stats.json`: Statistical information about features\n\n")

            f.write("## Usage\n")
            f.write("To load this feature store:\n\n")
            f.write("```python\n")
            f.write("import pandas as pd\n")
            f.write("import pickle\n\n")
            f.write("# Load features\n")
            f.write("X_train = pd.read_parquet('X_train.parquet')\n")
            f.write("y_train = pd.read_parquet('y_train.parquet')['target']\n\n")
            f.write("# Load embedders\n")
            f.write("with open('embedders.pkl', 'rb') as f:\n")
            f.write("    embedders = pickle.load(f)\n")
            f.write("```\n")

        print(f"Feature store saved to {feature_store_dir}")
        return feature_store_dir