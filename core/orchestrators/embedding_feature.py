# core/orchestrators/embedding_feature.py

import sqlite3
import pandas as pd
import json
import numpy as np
import os
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

from utils.env_setup import AppConfig
from utils.cache_manager import CacheManager
from model_training.features.horse_embedding import HorseEmbedding
from model_training.features.jockey_embedding import JockeyEmbedding
from model_training.features.course_embedding import CourseEmbedding
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
        self.horse_embedder = HorseEmbedding(embedding_dim=self.embedding_dim)
        self.jockey_embedder = JockeyEmbedding(embedding_dim=self.embedding_dim)
        self.course_embedder = CourseEmbedding(embedding_dim=self.embedding_dim)
        self.couple_embedder = CoupleEmbedding(embedding_dim=self.embedding_dim)

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
        print(f"  - Embedding dimension: {self.embedding_dim}")

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
        # Generate cache key
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
            try:
                cached_data = self.cache_manager.load_dataframe(cache_key)
                if cached_data is not None:
                    print("Using cached historical race data...")
                    return cached_data
            except Exception as e:
                print(f"Warning: Could not load from cache: {str(e)}. Loading from database...")

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

        # Cache the result
        if use_cache:
            try:
                self.cache_manager.save_dataframe(expanded_df, cache_key)
            except Exception as e:
                print(f"Warning: Could not save to cache: {str(e)}")

        return expanded_df

    def _expand_participants(self, df_races):
        """
        Expand the participants JSON into individual rows.

        Args:
            df_races: DataFrame with race data and JSON participants

        Returns:
            Expanded DataFrame with one row per participant
        """
        race_dfs = []

        for _, race in df_races.iterrows():
            try:
                if pd.isna(race['participants']):
                    continue

                participants = json.loads(race['participants'])

                if not participants:
                    continue

                # Create DataFrame for this race's participants
                race_df = pd.DataFrame(participants)

                # Add race information to each participant row
                for col in df_races.columns:
                    if col != 'participants' and col != 'ordre_arrivee':
                        race_df[col] = race[col]

                # Add result information if available
                if 'ordre_arrivee' in race and race['ordre_arrivee'] and not pd.isna(race['ordre_arrivee']):
                    try:
                        results = json.loads(race['ordre_arrivee'])
                        # Create a mapping of horse IDs to final positions
                        id_to_position = {res['idche']: res['narrivee'] for res in results}

                        # Add a column for the final position
                        race_df['final_position'] = race_df['idche'].map(id_to_position)
                    except json.JSONDecodeError:
                        pass

                race_dfs.append(race_df)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing race {race.get('comp', 'unknown')}: {str(e)}")
                continue

        # Combine all race DataFrames
        if race_dfs:
            combined_df = pd.concat(race_dfs, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        return combined_df

    def fit_embeddings(self, df, use_cache=True):
        """
        Fit embedding models on the data with caching.

        Args:
            df: DataFrame with race and participant data
            use_cache: Whether to use cached embeddings if available

        Returns:
            self for method chaining
        """
        # Generate cache key
        cache_params = {
            'race_count': df['comp'].nunique(),
            'participant_count': len(df),
            'columns': sorted(df.columns.tolist()),
            'embedding_dim': self.embedding_dim
        }
        cache_key = self._generate_cache_key('embeddings', cache_params)

        # Try to get from cache
        if use_cache:
            try:
                # FIX: Use cache_key directly as the cache_type
                cached_embeddings = self.cache_manager.load_dataframe(cache_key)
                if isinstance(cached_embeddings, dict) and 'embeddings_fitted' in cached_embeddings:
                    print("Using cached embedding models...")
                    self.embeddings_fitted = True
                    return self
            except Exception as e:
                print(f"Warning: Could not load embeddings from cache: {str(e)}")

        print("Fitting entity embeddings...")

        # Handle Course embeddings
        if 'idche' in df.columns:
            try:
                # Prepare for Course embedding - ensure numeric IDs
                df['idche'] = pd.to_numeric(df['idche'], errors='coerce').fillna(-1).astype(int)

                # Course data for horse-track interaction
                course_info = df[['comp', 'hippo', 'typec', 'dist', 'meteo', 'temperature']].drop_duplicates('comp')

                # Fit the course embedder for use with horses
                if len(course_info) > 5:
                    self.course_embedder.fit(course_info)

                print("Course embeddings prepared")
            except Exception as e:
                print(f"Warning: Could not prepare Course embeddings: {str(e)}")

        # Handle jockey embeddings
        if 'idJockey' in df.columns:
            try:
                # Prepare for jockey embedding
                df['idJockey'] = pd.to_numeric(df['idJockey'], errors='coerce').fillna(-1).astype(int)

                # Fit jockey embedder
                self.jockey_embedder.fit(df)
                print("Jockey embeddings fitted")
            except Exception as e:
                print(f"Warning: Could not fit jockey embeddings: {str(e)}")

        # Handle couple embeddings
        if 'idche' in df.columns and 'idJockey' in df.columns and 'final_position' in df.columns:
            try:
                # Train couple embeddings
                self.couple_embedder.train(df, target_col='final_position')
                print("Couple embeddings trained")
            except Exception as e:
                print(f"Warning: Could not train couple embeddings: {str(e)}")

        self.embeddings_fitted = True

        # Cache embeddings status
        if use_cache:
            try:
                embeddings_status = {'embeddings_fitted': True}
                # FIX: Pass the cache_key directly as the cache_type
                self.cache_manager.save_dataframe(embeddings_status, cache_key)
            except Exception as e:
                print(f"Warning: Could not cache embedding status: {str(e)}")

        return self

    def apply_embeddings(self, df, use_cache=True):
        """
        Apply fitted embeddings to the data with caching.

        Args:
            df: DataFrame with race and participant data
            use_cache: Whether to use cached transformations

        Returns:
            DataFrame with embedded features added
        """
        # Generate cache key
        cache_params = {
            'data_shape': df.shape,
            'data_columns': sorted(df.columns.tolist()),
            'embedding_dim': self.embedding_dim,
            'horse_count': df['idche'].nunique() if 'idche' in df.columns else 0,
            'jockey_count': df['idJockey'].nunique() if 'idJockey' in df.columns else 0
        }
        cache_key = self._generate_cache_key('embedded_features', cache_params)

        # Try to get from cache
        if use_cache:
            try:
                cached_df = self.cache_manager.load_dataframe(cache_key)
                if cached_df is not None and isinstance(cached_df, pd.DataFrame):
                    print("Using cached embedded features...")
                    return cached_df
            except Exception as e:
                print(f"Warning: Could not load embedded features from cache: {str(e)}")

        if not self.embeddings_fitted:
            print("Embeddings not fitted yet. Fitting now...")
            self.fit_embeddings(df, use_cache=use_cache)

        print("Applying entity embeddings...")

        # Make a copy to avoid modifying the original
        embedded_df = df.copy()

        # Convert ID columns to integer type
        id_columns = ['idche', 'idJockey', 'idEntraineur']
        for col in id_columns:
            if col in embedded_df.columns:
                embedded_df[col] = pd.to_numeric(embedded_df[col], errors='coerce').fillna(-1).astype(int)

        # Apply horse embeddings
        if 'idche' in embedded_df.columns:
            try:
                # Generate horse embeddings
                embeddings_dict = self.horse_embedder.generate_embeddings(embedded_df)

                # Add embedding vectors as features
                for i in range(min(self.embedding_dim, 16)):  # Horse embedder uses at most 16 dimensions
                    embedded_df[f'horse_emb_{i}'] = embedded_df['idche'].map(
                        lambda x: embeddings_dict.get(x, np.zeros(16))[i] if pd.notnull(x) else 0
                    )

                print("Added horse embeddings")
            except Exception as e:
                print(f"Warning: Could not apply horse embeddings: {str(e)}")

        # Apply jockey embeddings
        if 'idJockey' in embedded_df.columns:
            try:
                # Transform through batch process
                jockey_features = self.jockey_embedder.transform_batch(embedded_df)

                # Add jockey embedding columns
                for i in range(self.embedding_dim):
                    col_name = f'jockey_emb_{i}'
                    if col_name in jockey_features.columns:
                        embedded_df[col_name] = jockey_features[col_name]

                print("Added jockey embeddings")
            except Exception as e:
                print(f"Warning: Could not apply jockey embeddings: {str(e)}")

        # Apply couple embeddings
        if 'idche' in embedded_df.columns and 'idJockey' in embedded_df.columns:
            try:
                # Add embeddings for horse-jockey combinations
                couple_features = self.couple_embedder.transform_df(embedded_df)

                # Add couple embedding columns
                for i in range(min(self.embedding_dim, 8)):  # Use at most 8 dimensions for couples
                    col_name = f'couple_emb_{i}'
                    if col_name in couple_features.columns:
                        embedded_df[col_name] = couple_features[col_name]

                print("Added couple embeddings")
            except Exception as e:
                print(f"Warning: Could not apply couple embeddings: {str(e)}")

        # Apply course embeddings if we have enough race data
        if 'comp' in embedded_df.columns:
            try:
                # Get unique courses
                courses = embedded_df[['comp', 'hippo', 'typec', 'dist', 'meteo', 'temperature']].drop_duplicates(
                    'comp')

                if len(courses) > 0:
                    # Transform courses
                    course_embeddings = self.course_embedder.transform(courses)

                    # Create a mapping from comp to embedding
                    course_embedding_dict = {}
                    for i, (_, course) in enumerate(courses.iterrows()):
                        course_embedding_dict[course['comp']] = course_embeddings[i]

                    # Add course embedding columns
                    for i in range(min(self.embedding_dim, course_embeddings.shape[1])):
                        embedded_df[f'course_emb_{i}'] = embedded_df['comp'].map(
                            lambda x: course_embedding_dict.get(x, np.zeros(self.embedding_dim))[
                                i] if x in course_embedding_dict else 0
                        )

                    print("Added course embeddings")
            except Exception as e:
                print(f"Warning: Could not apply course embeddings: {str(e)}")

        # Cache the transformed DataFrame
        if use_cache:
            try:
                self.cache_manager.save_dataframe(embedded_df, cache_key)
            except Exception as e:
                print(f"Warning: Could not cache embedded features: {str(e)}")

        return embedded_df

    def prepare_features(self, df, use_cache=True):
        """
        Prepare features for training, applying embeddings and handling categorical variables.

        Args:
            df: DataFrame with expanded participants data
            use_cache: Whether to use cached results

        Returns:
            DataFrame with processed features and embeddings
        """
        # Generate cache key
        cache_params = {
            'data_shape': df.shape,
            'data_columns': sorted(df.columns.tolist()),
            'embedding_dim': self.embedding_dim
        }
        cache_key = self._generate_cache_key('prepared_features', cache_params)

        # Try to get from cache
        if use_cache:
            try:
                # FIX: Use cache_key directly as the cache_type, remove 'features' parameter
                cached_df = self.cache_manager.load_dataframe(cache_key)
                if cached_df is not None:
                    print("Using cached prepared features...")
                    return cached_df
            except Exception as e:
                print(f"Warning: Could not load prepared features from cache: {str(e)}")

        print("Preparing features...")

        # Make a copy to avoid modifying the original
        processed_df = df.copy()

        # Handle missing numerical values
        numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(0)

        # Record preprocessing parameters for inference
        self.preprocessing_params['numeric_columns'] = numeric_cols.tolist()

        # Handle categorical features
        categorical_cols = ['corde', 'meteo', 'natpis', 'pistegp']

        for col in categorical_cols:
            if col in processed_df.columns:
                # Convert to category type
                processed_df[col] = processed_df[col].astype('category')

        self.preprocessing_params['categorical_columns'] = categorical_cols

        # Convert date to datetime and extract useful features
        if 'jour' in processed_df.columns:
            processed_df['jour'] = pd.to_datetime(processed_df['jour'], errors='coerce')
            processed_df['year'] = processed_df['jour'].dt.year
            processed_df['month'] = processed_df['jour'].dt.month
            processed_df['dayofweek'] = processed_df['jour'].dt.dayofweek

        # Apply entity embeddings
        processed_df = self.apply_embeddings(processed_df, use_cache=use_cache)

        # Cache the result
        if use_cache:
            try:
                # FIX: Pass cache_key directly as the cache_type, remove 'features' parameter
                self.cache_manager.save_dataframe(processed_df, cache_key)
            except Exception as e:
                print(f"Warning: Could not cache prepared features: {str(e)}")

        return processed_df

    def prepare_target_variable(self, df, target_column='final_position', task_type=None):
        """
        Prepare the target variable for training.

        Args:
            df: DataFrame with features and target
            target_column: Name of the target column
            task_type: 'regression', 'classification', or 'ranking', if None uses default from config

        Returns:
            Processed target variable
        """
        # Use default task type from config if not specified
        if task_type is None:
            task_type = self.config.get_default_task_type()

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        if task_type == 'regression':
            # For regression (e.g., predicting finish position as a number)
            # Convert positions to numeric, handling non-numeric values
            y = pd.to_numeric(df[target_column], errors='coerce')

            # Fill NaN values with a high number (effectively placing non-finishers last)
            max_pos = y.max()
            y.fillna(max_pos + 1, inplace=True)

        elif task_type == 'classification':
            # For classification (e.g., predicting win/place/show)
            # First, convert positions to categorical
            y = df[target_column].astype('str')

            # Map positions to categories:
            # 1 = Win, 2-3 = Place, 4+ = Other
            def categorize_position(pos):
                try:
                    pos_num = int(pos)
                    if pos_num == 1:
                        return 'win'
                    elif pos_num <= 3:
                        return 'place'
                    else:
                        return 'other'
                except (ValueError, TypeError):
                    return 'other'  # Non-numeric positions (DNF, etc.)

            y = y.apply(categorize_position)

        elif task_type == 'ranking':
            # For ranking models (e.g., learning to rank horses within a race)
            # First, convert positions to numeric
            y = pd.to_numeric(df[target_column], errors='coerce')

            # Group by race ID to get the race context
            race_groups = df.groupby('comp')

            # Prepare race context dataframe with proper ranking
            race_contexts = []

            for comp, group in race_groups:
                # Rank horses within the race (lower position is better)
                group = group.copy()
                group['rank'] = group[target_column].rank(method='min', na_option='bottom')
                race_contexts.append(group)

            # Combine all groups back
            ranked_df = pd.concat(race_contexts)

            # Return both the ranking and the grouped dataframe for specialized ranking models
            return ranked_df['rank'], ranked_df

        else:
            raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression', 'classification', or 'ranking'")

        return y

    def prepare_training_dataset(self, df, target_column='final_position', task_type=None, race_group_split=False):
        """
        Prepare the final dataset for training.

        Args:
            df: DataFrame with all data
            target_column: Column to use as the target variable
            task_type: 'regression', 'classification', or 'ranking', if None uses default from config
            race_group_split: Whether to split by race groups (entire races go to train/val/test)

        Returns:
            X: Features DataFrame
            y: Target Series
            (Optional) groups: Race identifiers for grouped cross-validation
        """
        # Use default task type from config if not specified
        if task_type is None:
            task_type = self.config.get_default_task_type()

        # Drop rows with missing target values
        training_df = df.dropna(subset=[target_column])

        # Prepare the target variable
        if task_type == 'ranking':
            y, ranked_df = self.prepare_target_variable(training_df, target_column, task_type)
            training_df = ranked_df  # Use the ranked dataframe
        else:
            y = self.prepare_target_variable(training_df, target_column, task_type)

        # Select feature columns (excluding target and non-feature columns)
        exclude_cols = [
            target_column, 'rank', 'comp', 'idche', 'cheval', 'ordre_arrivee', 'idJockey', 'idEntraineur',
            'participants', 'created_at', 'jour', 'prix', 'reun', 'final_position', 'quinte',
            'proprietaire', 'musiqueche', 'musiquejoc'
        ]

        # Store excluded columns for reference
        self.preprocessing_params['excluded_columns'] = exclude_cols

        # Remove columns that are in the exclude list and exist in the DataFrame
        feature_cols = [col for col in training_df.columns if col not in exclude_cols or col not in training_df.columns]
        X = training_df[feature_cols]

        # Store feature columns for consistency in inference
        self.preprocessing_params['feature_columns'] = feature_cols

        # Return groups for group-based splitting if requested
        if race_group_split:
            groups = training_df['comp']
            return X, y, groups

        return X, y

    def split_dataset(self, X, y, test_size=0.2, val_size=0.1, random_state=42, groups=None):
        """
        Split the dataset into training, validation, and testing sets.

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            groups: Optional group labels for group-based splitting

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        from sklearn.model_selection import train_test_split, GroupShuffleSplit

        if groups is not None:
            # Group-based splitting (entire races go to train/val/test)
            # First split: training+validation vs test
            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_val_idx, test_idx = next(gss_test.split(X, y, groups))

            X_trainval, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
            y_trainval, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
            groups_trainval = groups.iloc[train_val_idx]

            # Second split: training vs validation
            # Adjusted val_size to be a percentage of the trainval set
            val_adjusted = val_size / (1 - test_size)

            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_adjusted, random_state=random_state)
            train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups_trainval))

            X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
            y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

        else:
            # Regular random splitting
            # First split: training+validation vs test
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Second split: training vs validation
            # Adjusted val_size to be a percentage of the trainval set
            val_adjusted = val_size / (1 - test_size)

            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=val_adjusted, random_state=random_state
            )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def run_pipeline(self, limit=None, race_filter=None, date_filter=None,
                     task_type=None, test_size=0.2, val_size=0.1,
                     race_group_split=False, random_state=42, embedding_dim=None,
                     use_cache=True):
        """
        Run the complete pipeline from data loading to training set preparation.

        Args:
            limit: Optional limit for races to load
            race_filter: Optional filter for specific race types
            date_filter: Optional date filter
            task_type: 'regression', 'classification', or 'ranking', if None uses default from config
            test_size: Proportion for test split
            val_size: Proportion for validation split
            race_group_split: Whether to split by race groups
            random_state: Random seed
            embedding_dim: Override default embedding dimension
            use_cache: Whether to use cached results when available

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Use default task type from config if not specified
        if task_type is None:
            task_type = self.config.get_default_task_type()

        # Update embedding dimension if specified
        if embedding_dim is not None and embedding_dim != self.embedding_dim:
            self.embedding_dim = embedding_dim
            self.embeddings_fitted = False

        # Generate cache key for the entire pipeline run
        pipeline_params = {
            'limit': limit,
            'race_filter': race_filter,
            'date_filter': date_filter,
            'task_type': task_type,
            'test_size': test_size,
            'val_size': val_size,
            'race_group_split': race_group_split,
            'random_state': random_state,
            'embedding_dim': self.embedding_dim
        }
        pipeline_key = self._generate_cache_key('pipeline_run', pipeline_params)

        # Try to get from cache
        if use_cache:
            try:
                cached_result = self.cache_manager.load_dataframe('pipeline_run', pipeline_key)
                if cached_result is not None:
                    print("Using cached pipeline results...")
                    return cached_result
            except Exception as e:
                print(f"Warning: Could not load pipeline results from cache: {str(e)}")

        # Load data
        df = self.load_historical_races(
            limit=limit,
            race_filter=race_filter,
            date_filter=date_filter,
            use_cache=use_cache
        )
        print(f"Loaded {len(df)} participant records from {df['comp'].nunique()} races")

        # Store pipeline parameters
        self.preprocessing_params.update({
            'pipeline_params': pipeline_params,
            'data_shape': df.shape,
            'race_count': df['comp'].nunique()
        })

        # Prepare features with embeddings
        features_df = self.prepare_features(df, use_cache=use_cache)

        # Prepare training dataset
        print("Preparing training dataset...")
        if race_group_split:
            X, y, groups = self.prepare_training_dataset(
                features_df, task_type=task_type, race_group_split=True
            )
            print(
                f"Dataset prepared with {X.shape[1]} features and {len(y)} samples across {len(groups.unique())} races")
        else:
            X, y = self.prepare_training_dataset(features_df, task_type=task_type)
            groups = None
            print(f"Dataset prepared with {X.shape[1]} features and {len(y)} samples")

        # Split for training, validation, and testing
        print("Splitting dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(
            X, y, test_size=test_size, val_size=val_size, random_state=random_state, groups=groups
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Get feature importance if applicable
        self._check_feature_correlation(X_train, y_train)

        # Cache the pipeline result
        if use_cache:
            try:
                result = (X_train, X_val, X_test, y_train, y_val, y_test)
                self.cache_manager.save_dataframe(result, 'pipeline_run', pipeline_key)
            except Exception as e:
                print(f"Warning: Could not cache pipeline results: {str(e)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _check_feature_correlation(self, X, y, top_n=10):
        """
        Check correlation between features and target.

        Args:
            X: Feature DataFrame
            y: Target Series
            top_n: Number of top correlated features to display
        """
        # Only proceed if y is numeric
        if not pd.api.types.is_numeric_dtype(y):
            return

        try:
            # Create a dataframe with features and target
            corr_df = X.copy()
            corr_df['target'] = y

            # Calculate correlation with target
            correlations = corr_df.corr()['target'].sort_values(ascending=False)

            # Print top features (excluding target itself)
            print("\nTop correlated features:")
            for i, (feature, corr) in enumerate(correlations[1:top_n + 1].items()):
                print(f"  {i + 1}. {feature}: {corr:.4f}")

            # Store feature importances
            self.preprocessing_params['feature_importances'] = correlations.to_dict()

        except Exception as e:
            print(f"Could not calculate feature correlations: {str(e)}")

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
                'course_embedder': self.course_embedder,
                'couple_embedder': self.couple_embedder
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

    def load_feature_store(self, feature_store_path=None, store_name=None):
        """
        Load a previously saved feature store.

        Args:
            feature_store_path: Path to the feature store directory
            store_name: Name of a specific feature store in the default directory

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, embedders, metadata)
        """
        # Determine the feature store path
        if feature_store_path is None:
            if store_name is None:
                # List available feature stores
                stores = self._list_feature_stores()
                if not stores:
                    raise ValueError("No feature stores found in the default directory. Please specify a path or name.")
                # Use the most recent one
                feature_store_path = os.path.join(self.feature_store_dir, stores[0])
                print(f"Using most recent feature store: {stores[0]}")
            else:
                # Find the store by name
                stores = self._list_feature_stores()
                matched_stores = [s for s in stores if store_name in s]
                if not matched_stores:
                    raise ValueError(f"No feature stores matching '{store_name}' found.")
                feature_store_path = os.path.join(self.feature_store_dir, matched_stores[0])

        print(f"Loading feature store from {feature_store_path}...")

        # Load features
        X_train = pd.read_parquet(os.path.join(feature_store_path, "X_train.parquet"))
        X_val = pd.read_parquet(os.path.join(feature_store_path, "X_val.parquet"))
        X_test = pd.read_parquet(os.path.join(feature_store_path, "X_test.parquet"))

        # Load targets
        y_train = pd.read_parquet(os.path.join(feature_store_path, "y_train.parquet"))['target']
        y_val = pd.read_parquet(os.path.join(feature_store_path, "y_val.parquet"))['target']
        y_test = pd.read_parquet(os.path.join(feature_store_path, "y_test.parquet"))['target']

        # Load embedders
        with open(os.path.join(feature_store_path, "embedders.pkl"), 'rb') as f:
            embedders = pickle.load(f)

        # Load metadata
        with open(os.path.join(feature_store_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)

        # Update the orchestrator with the loaded embedders
        self.horse_embedder = embedders.get('horse_embedder', self.horse_embedder)
        self.jockey_embedder = embedders.get('jockey_embedder', self.jockey_embedder)
        self.course_embedder = embedders.get('course_embedder', self.course_embedder)
        self.couple_embedder = embedders.get('couple_embedder', self.couple_embedder)
        self.embeddings_fitted = True

        # Update preprocessing parameters
        if 'preprocessing' in metadata:
            self.preprocessing_params = metadata['preprocessing']

        if hasattr(self.horse_embedder, 'embedding_dim'):
            self.embedding_dim = self.horse_embedder.embedding_dim

        print("Feature store loaded successfully.")
        return X_train, X_val, X_test, y_train, y_val, y_test, embedders, metadata

    def _list_feature_stores(self):
        """
        List available feature stores in the configured directory.

        Returns:
            List of feature store names, sorted by most recent first
        """
        if not os.path.exists(self.feature_store_dir):
            return []

        # Get all feature store directories
        stores = [d for d in os.listdir(self.feature_store_dir)
                  if os.path.isdir(os.path.join(self.feature_store_dir, d))
                  and d.startswith('feature_store_')]

        # Sort by timestamp (most recent first)
        stores.sort(reverse=True)

        return stores

    def clear_cache(self):
        """
        Clear all cached data.
        """
        try:
            # Call appropriate method based on CacheManager implementation
            self.cache_manager.clear_cache()
            print("Cache cleared successfully.")
        except Exception as e:
            print(f"Warning: Could not clear cache: {str(e)}")