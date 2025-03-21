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
from sklearn.preprocessing import StandardScaler
from utils.env_setup import AppConfig
from utils.cache_manager import CacheManager
from model_training.features.horse_embedding import HorseEmbedding
from model_training.features.jockey_embedding import JockeyEmbedding
from model_training.features.course_embedding import CourseEmbedding
from model_training.features.couple_embedding import CoupleEmbedding


class FeatureEmbeddingOrchestrator:
    """
    Orchestrator for loading historical race data, applying entity embeddings,
    and preparing data for model training. Consolidates data preparation functionality
    from both embedding and model training components.
    """

    def __init__(self, sqlite_path=None, embedding_dim=None, cache_dir=None, feature_store_dir=None,
                 sequence_length=5, verbose=False):
        """
        Initialize the orchestrator with embedding models and caching.

        Args:
            sqlite_path: Path to SQLite database, if None uses default from config
            embedding_dim: Dimension size for entity embeddings, if None uses default from config
            cache_dir: Directory to store cache files, if None uses default from config
            feature_store_dir: Directory to store feature stores, if None uses default from config
            sequence_length: Default sequence length for LSTM data preparation
            verbose: Whether to print verbose output
        """
        # Load application configuration
        self.config = AppConfig()

        # Set paths from config or arguments
        self.sqlite_path = sqlite_path or self.config.get_active_db_path()
        self.cache_dir = cache_dir or self.config.get_cache_dir()
        self.feature_store_dir = feature_store_dir or self.config.get_feature_store_dir()

        # Set embedding dimension from config or argument
        self.embedding_dim = embedding_dim or self.config.get_default_embedding_dim()

        # Set sequence length for LSTM data
        self.sequence_length = sequence_length

        # Set verbosity
        self.verbose = verbose

        # Initialize caching manager
        self.cache_manager = CacheManager()

        # Initialize embedding models (will be fitted later)
        self.horse_embedder = HorseEmbedding(embedding_dim=self.embedding_dim)
        self.jockey_embedder = JockeyEmbedding(embedding_dim=self.embedding_dim)
        self.course_embedder = CourseEmbedding(embedding_dim=10)
        self.couple_embedder = CoupleEmbedding(embedding_dim=self.embedding_dim)

        # Track whether embeddings have been fitted
        self.embeddings_fitted = False

        # Store preprocessing parameters
        self.preprocessing_params = {
            'sequence_length': sequence_length
        }

        self.target_info = {
            'column': 'final_position',
            'type': 'regression'  # Options: 'regression', 'classification', 'ranking'
        }

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.feature_store_dir, exist_ok=True)

        if self.verbose:
            print(f"Orchestrator initialized with:")
            print(f"  - SQLite path: {self.sqlite_path}")
            print(f"  - Cache directory: {self.cache_dir}")
            print(f"  - Feature store directory: {self.feature_store_dir}")
            print(f"  - Embedding dimension: {self.embedding_dim}")
            print(f"  - Sequence length: {self.sequence_length}")

    def log_info(self, message):
        """Simple logging method for backward compatibility."""
        if hasattr(self, 'verbose') and self.verbose:
            print(message)

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
                course_info = df[['comp', 'hippo', 'typec', 'dist', 'meteo', 'temperature', 'natpis']].drop_duplicates(
                    'comp')

                # Fit the course embedder for use with horses
                if len(course_info) > 5:
                    self.course_embedder.fit(course_info)

                print("Course embeddings prepared")
            except Exception as e:
                print(f"Warning: Could not prepare Course embeddings: {str(e)}")
                import traceback
                traceback.print_exc()

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

    def drop_embedded_raw_features(self, df, keep_identifiers=False):
        """
        Drop raw features that have already been converted to embeddings.

        Args:
            df: DataFrame with feature columns
            keep_identifiers: Whether to keep ID columns (idche, idJockey, etc.) which may be
                          needed for further processing

        Returns:
            DataFrame with raw embedded features removed
        """
        # Make a copy to avoid modifying the input
        clean_df = df.copy()

        # 1. Check which types of embeddings exist in the dataframe
        has_horse_embeddings = any(col.startswith('horse_emb_') for col in clean_df.columns)
        has_jockey_embeddings = any(col.startswith('jockey_emb_') for col in clean_df.columns)
        has_couple_embeddings = any(col.startswith('couple_emb_') for col in clean_df.columns)
        has_course_embeddings = any(col.startswith('course_emb_') for col in clean_df.columns)

        # 2. Create lists of raw features to drop for each embedding type
        columns_to_drop = []

        # Horse-related raw features
        if has_horse_embeddings:
            horse_raw_features = [
                # Performance statistics
                'victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere',
                'pourcVictChevalHippo', 'pourcPlaceChevalHippo',
                # Derived ratios
                'ratio_victoires', 'ratio_places', 'gains_par_course', 'perf_cheval_hippo',
                # Musique-derived features
                'musiqueche'
            ]

            # Add all che_* columns (global, weighted, bytype)
            che_columns = [col for col in clean_df.columns if
                           col.startswith(('che_global_', 'che_weighted_', 'che_bytype_'))]
            horse_raw_features.extend(che_columns)

            columns_to_drop.extend(horse_raw_features)

        # Jockey-related raw features
        if has_jockey_embeddings:
            jockey_raw_features = [
                'pourcVictJockHippo', 'pourcPlaceJockHippo',
                'musiquejoc'
            ]

            # Add all joc_* columns
            joc_columns = [col for col in clean_df.columns if
                           col.startswith(('joc_global_', 'joc_weighted_', 'joc_bytype_'))]
            jockey_raw_features.extend(joc_columns)

            columns_to_drop.extend(jockey_raw_features)

        # Couple-related raw features
        if has_couple_embeddings:
            couple_raw_features = [
                'nbVictCouple', 'nbPlaceCouple', 'nbCourseCouple', 'TxVictCouple',
                'efficacite_couple', 'regularite_couple', 'progression_couple'
            ]
            columns_to_drop.extend(couple_raw_features)

        # Course-related raw features
        if has_course_embeddings:
            # These are already captured in course embeddings but might be kept as direct features
            # based on your model requirements. If you're certain they're redundant, uncomment:
            # course_raw_features = ['hippo', 'natpis', 'dist', 'meteo', 'temperature', 'forceVent', 'directionVent']
            # columns_to_drop.extend(course_raw_features)
            pass

        # 3. Handle identifiers
        identifiers = ['idche', 'idJockey', 'idEntraineur', 'proprietaire', 'comp', 'couple_id']

        if not keep_identifiers:
            columns_to_drop.extend(identifiers)

        # 4. Add other metadata columns that should generally be excluded from training
        metadata_columns = [
            'cheval', 'ordre_arrivee', 'participants', 'created_at', 'jour',
            'prix', 'reun', 'quinte', 'cl', 'narrivee'
        ]
        columns_to_drop.extend(metadata_columns)

        # 5. Only drop columns that actually exist in the dataframe
        columns_to_drop = [col for col in columns_to_drop if col in clean_df.columns]

        # 6. Drop the columns and save the list for reference
        clean_df = clean_df.drop(columns=columns_to_drop)

        # Save the list of dropped columns for reference
        self.preprocessing_params['dropped_raw_features'] = columns_to_drop

        self.log_info(f"Dropped {len(columns_to_drop)} raw feature columns that were already embedded")

        return clean_df

    def apply_embeddings(self, df, use_cache=True, clean_after_embedding=True, keep_identifiers=False):
        """
        Apply fitted embeddings to the data with caching.

        Args:
            df: DataFrame with race and participant data
            use_cache: Whether to use cached transformations
            clean_after_embedding: Whether to drop raw features after embedding
            keep_identifiers: Whether to keep identifier columns even when cleaning

        Returns:
            DataFrame with embedded features added (and raw features optionally removed)
        """
        # Generate cache key
        cache_params = {
            'data_shape': df.shape,
            'data_columns': sorted(df.columns.tolist()),
            'embedding_dim': self.embedding_dim,
            'horse_count': df['idche'].nunique() if 'idche' in df.columns else 0,
            'jockey_count': df['idJockey'].nunique() if 'idJockey' in df.columns else 0,
            'clean_after_embedding': clean_after_embedding
        }
        cache_key = self._generate_cache_key('embedded_features', cache_params)

        # Try to get from cache
        if use_cache:
            try:
                cached_df = self.cache_manager.load_dataframe(cache_key)
                if cached_df is not None and isinstance(cached_df, pd.DataFrame):
                    self.log_info("Using cached embedded features...")
                    return cached_df
            except Exception as e:
                self.log_info(f"Warning: Could not load embedded features from cache: {str(e)}")

        if not self.embeddings_fitted:
            self.log_info("Embeddings not fitted yet. Fitting now...")
            self.fit_embeddings(df, use_cache=use_cache)

        self.log_info("Applying entity embeddings...")

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

                self.log_info("Added horse embeddings")
            except Exception as e:
                self.log_info(f"Warning: Could not apply horse embeddings: {str(e)}")

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

                self.log_info("Added jockey embeddings")
            except Exception as e:
                self.log_info(f"Warning: Could not apply jockey embeddings: {str(e)}")

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

                self.log_info("Added couple embeddings")
            except Exception as e:
                self.log_info(f"Warning: Could not apply couple embeddings: {str(e)}")

        # Apply course embeddings if we have enough race data
        if 'comp' in embedded_df.columns:
            try:
                # Get unique courses
                courses = embedded_df[
                    ['comp', 'hippo', 'typec', 'dist', 'meteo', 'temperature', 'natpis']].drop_duplicates('comp')
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
                            lambda x: course_embedding_dict.get(x, np.zeros(self.embedding_dim))[i]
                            if x in course_embedding_dict else 0
                        )

                    self.log_info("Added course embeddings")
            except Exception as e:
                self.log_info(f"Warning: Could not apply course embeddings: {str(e)}")

        # Clean embedded data if requested
        if clean_after_embedding:
            embedded_df = self.drop_embedded_raw_features(embedded_df, keep_identifiers=keep_identifiers)

        # Cache the transformed DataFrame
        if use_cache:
            try:
                self.cache_manager.save_dataframe(embedded_df, cache_key)
            except Exception as e:
                self.log_info(f"Warning: Could not cache embedded features: {str(e)}")

        return embedded_df

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

        # Ensure raw features have been dropped (in case it wasn't done after embedding)
        if any(col.startswith(('horse_emb_', 'jockey_emb_', 'couple_emb_', 'course_emb_')) for col in
               training_df.columns):
            # Keep identifiers at this stage if we need them for grouping
            keep_ids = race_group_split
            training_df = self.drop_embedded_raw_features(training_df, keep_identifiers=keep_ids)

        # Prepare the target variable
        if task_type == 'ranking':
            y, ranked_df = self.prepare_target_variable(training_df, target_column, task_type)
            training_df = ranked_df  # Use the ranked dataframe
        else:
            y = self.prepare_target_variable(training_df, target_column, task_type)

        # Select feature columns (excluding target and identifier columns)
        exclude_cols = [target_column]

        # Always exclude these columns, even if we kept identifiers earlier
        exclude_cols.extend(['comp', 'rank', 'idche', 'idJockey', 'idEntraineur', 'couple_id'])

        # Select all remaining columns for training
        feature_cols = [col for col in training_df.columns if col not in exclude_cols]
        X_raw = training_df[feature_cols]

        # Clean the feature dataframe for training
        X = self.clean_feature_dataframe(X_raw)

        # Store feature columns for consistency in inference
        self.preprocessing_params['feature_columns'] = X.columns.tolist()

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
            self.log_info("Cannot calculate correlations: target is not numeric")
            return

        try:
            # Clean the feature dataframe
            self.log_info("Cleaning feature dataframe for correlation analysis...")
            corr_df = self.clean_feature_dataframe(X)

            # Add target to the correlation DataFrame
            corr_df['target'] = y

            # Calculate correlation with target
            self.log_info("Calculating correlations with target...")
            correlations = corr_df.corr()['target'].sort_values(ascending=False)

            # Print top features (excluding target itself)
            self.log_info("\nTop correlated features:")
            for i, (feature, corr) in enumerate(correlations[1:top_n + 1].items()):
                self.log_info(f"  {i + 1}. {feature}: {corr:.4f}")

            # Store feature importances
            self.preprocessing_params['feature_importances'] = correlations.to_dict()

        except Exception as e:
            self.log_info(f"Could not calculate feature correlations: {str(e)}")
            if self.verbose:
                import traceback
                self.log_info(traceback.format_exc())

    def clean_feature_dataframe(self, df):
        """
        Clean a dataframe to ensure all features can be used for numerical operations.

        Args:
            df: DataFrame with features

        Returns:
            Cleaned DataFrame with properly handled features
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()

        # List of columns with known empty string issues
        problematic_columns = ['poidmont', 'directionVent', 'nebulosite']

        for col in cleaned_df.columns:
            # Handle empty strings
            if cleaned_df[col].dtype == 'object':
                # Check if column has empty strings
                empty_count = (cleaned_df[col] == '').sum()
                if empty_count > 0:
                    self.log_info(f"Handling empty strings in column '{col}': {empty_count} empties")

                    # Option 1: Convert empty strings to NaN, then fill with appropriate value
                    cleaned_df[col] = cleaned_df[col].replace('', np.nan)

                    # For known problematic columns, use domain-specific defaults
                    if col == 'poidmont':
                        # Weight carried - use median or mean of non-empty values
                        median_value = pd.to_numeric(cleaned_df[col], errors='coerce').median()
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(median_value)

                    elif col == 'directionVent':
                        # Wind direction - categorical, so use most common value or '0'
                        # Convert to categorical then to numeric code
                        cleaned_df[col] = cleaned_df[col].fillna('unknown')
                        # Create new column with categorical encoding
                        cleaned_df[f'{col}_code'] = cleaned_df[col].astype('category').cat.codes
                        # Drop original string column
                        cleaned_df = cleaned_df.drop(columns=[col])

                    elif col == 'nebulosite':
                        # Cloud cover - categorical, similar approach
                        cleaned_df[col] = cleaned_df[col].fillna('unknown')
                        cleaned_df[f'{col}_code'] = cleaned_df[col].astype('category').cat.codes
                        cleaned_df = cleaned_df.drop(columns=[col])

                    else:
                        # For other columns, convert to numeric and fill NaN with 0
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(0)

            # Handle other non-numeric columns
            elif not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                # If categorical, convert to category codes
                if pd.api.types.is_categorical_dtype(cleaned_df[col]):
                    self.log_info(f"Converting categorical column '{col}' to codes")
                    cleaned_df[col] = cleaned_df[col].cat.codes
                # For other types, try converting to numeric or drop
                else:
                    try:
                        self.log_info(f"Attempting to convert column '{col}' to numeric")
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

                        # If conversion resulted in all NaNs, drop the column
                        if cleaned_df[col].isna().all():
                            self.log_info(f"Dropping column '{col}' - all values are NaN after conversion")
                            cleaned_df = cleaned_df.drop(columns=[col])
                    except:
                        self.log_info(f"Cannot convert column '{col}' to numeric - dropping")
                        cleaned_df = cleaned_df.drop(columns=[col])

        # Fill any remaining NaN values
        cleaned_df = cleaned_df.fillna(0)

        return cleaned_df

    def run_pipeline(self, limit=None, race_filter=None, date_filter=None,
                     task_type=None, test_size=0.2, val_size=0.1,
                     race_group_split=False, random_state=42, embedding_dim=None,
                     use_cache=True, clean_embeddings=True):
        """
        Run the complete pipeline from data loading to training set preparation.

        Args:
            # Your existing arguments...
            clean_embeddings: Whether to drop raw features after embedding

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
            'embedding_dim': self.embedding_dim,
            'clean_embeddings': clean_embeddings
        }
        pipeline_key = self._generate_cache_key('pipeline_run', pipeline_params)

        # Try to get from cache
        if use_cache:
            try:
                cached_result = self.cache_manager.load_dataframe(pipeline_key)
                if cached_result is not None:
                    self.log_info("Using cached pipeline results...")
                    return cached_result
            except Exception as e:
                self.log_info(f"Warning: Could not load pipeline results from cache: {str(e)}")

        # Load data
        df = self.load_historical_races(
            limit=limit,
            race_filter=race_filter,
            date_filter=date_filter,
            use_cache=use_cache
        )
        self.log_info(f"Loaded {len(df)} participant records from {df['comp'].nunique()} races")

        # Store pipeline parameters
        self.preprocessing_params.update({
            'pipeline_params': pipeline_params,
            'data_shape': df.shape,
            'race_count': df['comp'].nunique()
        })

        # Prepare features with embeddings and optional cleaning
        features_df = self.prepare_features(df, use_cache=use_cache)

        # Apply embeddings if not already applied
        if not any(col.startswith(('horse_emb_', 'jockey_emb_')) for col in features_df.columns):
            features_df = self.apply_embeddings(features_df,
                                                use_cache=use_cache,
                                                clean_after_embedding=clean_embeddings,
                                                keep_identifiers=race_group_split)
        elif clean_embeddings:
            # If embeddings already exist but we want to clean
            features_df = self.drop_embedded_raw_features(features_df,
                                                          keep_identifiers=race_group_split)

        # Prepare training dataset
        self.log_info("Preparing training dataset...")
        if race_group_split:
            X, y, groups = self.prepare_training_dataset(
                features_df, task_type=task_type, race_group_split=True
            )
            self.log_info(
                f"Dataset prepared with {X.shape[1]} features and {len(y)} samples across {len(groups.unique())} races")
        else:
            X, y = self.prepare_training_dataset(features_df, task_type=task_type)
            groups = None
            self.log_info(f"Dataset prepared with {X.shape[1]} features and {len(y)} samples")

        # Split for training, validation, and testing
        self.log_info("Splitting dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(
            X, y, test_size=test_size, val_size=val_size, random_state=random_state, groups=groups
        )

        self.log_info(f"Training set: {X_train.shape[0]} samples")
        self.log_info(f"Validation set: {X_val.shape[0]} samples")
        self.log_info(f"Test set: {X_test.shape[0]} samples")

        # Get feature importance if applicable
        self._check_feature_correlation(X_train, y_train)

        # Cache the pipeline result
        if use_cache:
            try:
                result = (X_train, X_val, X_test, y_train, y_val, y_test)
                self.cache_manager.save_dataframe(result, pipeline_key)
            except Exception as e:
                self.log_info(f"Warning: Could not cache pipeline results: {str(e)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    # CONSOLIDATED METHODS FROM train_model.py

    def load_or_prepare_data(self, use_cache=True, limit=None, race_filter=None, date_filter=None):
        """
        Load or prepare data for training models.

        Args:
            use_cache: Whether to use cached data if available
            limit: Maximum number of races to load
            race_filter: Filter by race type
            date_filter: SQL-style date filter

        Returns:
            Tuple of (processed_features_df, static_features_df)
        """
        # Generate cache key for consistent results
        cache_params = {
            'limit': limit,
            'race_filter': race_filter,
            'date_filter': date_filter,
            'embedding_dim': self.embedding_dim
        }
        cache_key = self._generate_cache_key('training_data', cache_params)

        # Try to get from cache
        if use_cache:
            try:
                cached_data = self.cache_manager.load_dataframe(cache_key)
                if cached_data is not None:
                    self.log_info("Found cached training data")
                    df_features = cached_data
                    # Extract static features for RF with new approach
                    static_columns = [col for col in df_features.columns
                                      if not col.startswith(('comp', 'idche', 'id', 'cheval', 'ordre_arrivee'))]
                    static_features_df = df_features[static_columns].astype(float)
                    return df_features, static_features_df
            except Exception as e:
                self.log_info(f"Warning: Could not load from cache: {str(e)}. Loading fresh data...")

        self.log_info(f"Loading historical race data...")

        # Load raw data
        df_historical = self.load_historical_races(
            limit=limit,
            race_filter=race_filter,
            date_filter=date_filter,
            use_cache=use_cache
        )

        # Process and embed features
        df_features = self.prepare_features(df_historical, use_cache=use_cache)
        df_features = self.apply_embeddings(df_features, use_cache=use_cache,
                                            clean_after_embedding=True, keep_identifiers=True)

        # Extract static features for models that need them
        static_columns = [col for col in df_features.columns
                          if not col.startswith(('comp', 'idche', 'id', 'cheval', 'ordre_arrivee'))]
        static_features_df = df_features[static_columns].astype(float)

        # Cache the processed data
        if use_cache:
            try:
                self.cache_manager.save_dataframe(df_features, cache_key)
            except Exception as e:
                self.log_info(f"Warning: Could not cache processed data: {str(e)}")

        return df_features, static_features_df

    def prepare_sequence_data(self, df, sequence_length=None, step_size=1,
                              sequential_features=None, static_features=None):
        """
        Enhanced method for preparing sequential data for LSTM training that combines
        functionality from both the orchestrator and train_model.py.

        Args:
            df: DataFrame with race and participant data
            sequence_length: Number of races to include in each sequence (overrides instance value)
            step_size: Step size for sliding window
            sequential_features: List of features to use in sequences (None=auto-select)
            static_features: List of static features to include (None=auto-select)

        Returns:
            Tuple of (sequences, static_features, targets)
        """
        # Use instance sequence_length if not specified
        if sequence_length is None:
            sequence_length = self.sequence_length

        self.log_info(f"Preparing sequence data with length={sequence_length}")

        # If feature lists not provided, use sensible defaults
        if sequential_features is None:
            sequential_features = [
                'final_position', 'cotedirect', 'dist',
                # Include embeddings if available
                *[col for col in df.columns if col.startswith('horse_emb_')][:3],
                *[col for col in df.columns if col.startswith('jockey_emb_')][:3]
            ]
            # Add musique-derived features if available
            for prefix in ['che_global_', 'che_weighted_']:
                for feature in ['avg_pos', 'recent_perf', 'consistency', 'pct_top3']:
                    col = f"{prefix}{feature}"
                    if col in df.columns:
                        sequential_features.append(col)

        if static_features is None:
            static_features = [
                'age', 'temperature', 'natpis', 'typec', 'meteo', 'corde',
                # Include embeddings if available (ones not used in sequential)
                *[col for col in df.columns if col.startswith('couple_emb_')][:3],
                *[col for col in df.columns if col.startswith('course_emb_')][:3]
            ]

        # Ensure required columns exist
        required_cols = ['idche', 'jour']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing: {missing_cols}")

        # Filter to only include columns that exist in the dataframe
        sequential_features = [col for col in sequential_features if col in df.columns]
        static_features = [col for col in static_features if col in df.columns]

        self.log_info(f"Sequential features ({len(sequential_features)}): {sequential_features}")
        self.log_info(f"Static features ({len(static_features)}): {static_features}")

        # Sort by date
        df = df.sort_values(['idche', 'jour'])

        # Ensure all features are numeric
        for col in sequential_features + static_features:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Group by horse ID
        sequences = []
        static_feature_data = []
        targets = []

        # Get target column from configuration
        target_column = self.target_info.get('column', 'final_position')
        if target_column not in df.columns:
            if 'final_position' in df.columns:
                target_column = 'final_position'
            elif 'cl' in df.columns:
                target_column = 'cl'
            else:
                raise ValueError(f"Target column not found in DataFrame")

        for horse_id, horse_data in df.groupby('idche'):
            # Sort by date to ensure chronological order
            horse_races = horse_data.sort_values('jour')

            # Only process horses with enough races
            if len(horse_races) >= sequence_length + 1:
                # Get sequence features
                seq_features = horse_races[sequential_features].values.astype(np.float32)

                # Get last static features (from most recent race)
                static_feat = horse_races[static_features].iloc[-1].values.astype(np.float32)

                # Create sequences with sliding window
                for i in range(len(horse_races) - sequence_length):
                    # Sequence
                    seq = seq_features[i:i + sequence_length]
                    sequences.append(seq)

                    # Static features (same for all sequences of this horse)
                    static_feature_data.append(static_feat)

                    # Target (position in next race)
                    if target_column in horse_races.columns:
                        target_value = horse_races.iloc[i + sequence_length][target_column]
                        # Convert to float and handle non-numeric values
                        try:
                            target_value = float(target_value)
                        except (ValueError, TypeError):
                            # For DNF or other non-numeric results, use a high value
                            target_value = 99.0
                        targets.append(target_value)
                    else:
                        raise ValueError(f"Target column '{target_column}' not found")

        # Convert to numpy arrays
        if not sequences:
            raise ValueError("No valid sequences could be created. Check data quality and sequence length.")

        X_sequences = np.array(sequences, dtype=np.float32)
        X_static = np.array(static_feature_data, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        self.log_info(f"Created {len(sequences)} sequences from {df['idche'].nunique()} horses")
        self.log_info(f"Sequence shape: {X_sequences.shape}, Static shape: {X_static.shape}, Target shape: {y.shape}")

        # Store feature information
        self.preprocessing_params.update({
            'sequence_length': sequence_length,
            'step_size': step_size,
            'sequential_features': sequential_features,
            'static_features': static_features,
            'target_column': target_column
        })

        return X_sequences, X_static, y

    def create_hybrid_model(self, sequence_shape, static_shape, lstm_units=64, dropout_rate=0.2):
        """
        Create a hybrid model architecture with LSTM for sequential data and dense layers for static features.
        This consolidates the model creation logic from train_model.py.

        Args:
            sequence_shape: Shape of sequence input (seq_length, features)
            static_shape: Number of static features
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization

        Returns:
            Dictionary with model components
        """
        try:
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, concatenate
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            self.log_info("TensorFlow/Keras not available. Cannot create model.")
            return None

        # LSTM branch
        sequence_input = Input(shape=(sequence_shape[1], sequence_shape[2]), name='sequence_input')
        lstm = LSTM(lstm_units, return_sequences=False)(sequence_input)
        lstm = Dropout(dropout_rate)(lstm)

        # Static features branch
        static_input = Input(shape=(static_shape[1],), name='static_input')
        static_dense = Dense(32, activation='relu')(static_input)
        static_dense = Dropout(dropout_rate)(static_dense)

        # Combine branches
        combined = concatenate([lstm, static_dense])

        # Output layers
        dense = Dense(32, activation='relu')(combined)
        dense = Dropout(dropout_rate)(dense)
        output = Dense(1, activation='linear')(dense)

        # Create model
        model = Model(inputs=[sequence_input, static_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        return {
            'lstm': model,
            'sequence_input': sequence_input,
            'static_input': static_input
        }