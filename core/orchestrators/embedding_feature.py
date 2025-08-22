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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, concatenate, Lambda
from tensorflow.keras.optimizers import Adam


class FeatureEmbeddingOrchestrator:
    """
    Orchestrator for loading historical race data, applying entity embeddings,
    and preparing data for model training. Consolidates data preparation functionality
    from both embedding and model training components.
    """

    def __init__(self, sqlite_path=None, verbose=False):
        """
        Initialize the orchestrator with embedding models and caching.
        Most configuration now comes from config.yaml.

        Args:
            sqlite_path: Path to SQLite database, if None uses default from config
            verbose: Whether to print verbose output
        """
        # Load application configuration
        self.config = AppConfig()

        # Set path from config or argument
        self.sqlite_path = sqlite_path or self.config.get_active_db_path()

        # Load all configuration from config
        self.cache_dir = self.config.get_cache_dir()
        self.feature_store_dir = self.config.get_feature_store_dir()

        # Get feature configuration
        feature_config = self.config.get_features_config()
        self.embedding_dim = feature_config['embedding_dim']
        self.clean_after_embedding = feature_config.get('clean_after_embedding', True)
        self.keep_identifiers = feature_config.get('keep_identifiers', False)

        # Get LSTM configuration
        lstm_config = self.config.get_lstm_config()
        self.sequence_length = lstm_config['sequence_length']
        self.step_size = lstm_config['step_size']
        self.sequential_features = lstm_config.get('sequential_features', [])
        self.static_features = lstm_config.get('static_features', [])

        # Get dataset configuration
        dataset_config = self.config.get_dataset_config()
        self.test_size = dataset_config['test_size']
        self.val_size = dataset_config['val_size']
        self.random_state = dataset_config['random_state']

        # Cache setting
        self.use_cache = self.config.should_use_cache()

        # Set verbosity
        self.verbose = verbose

        # Initialize caching manager
        self.cache_manager = CacheManager()

        # Initialize embedding models (will be fitted later)
        self.horse_embedder = HorseEmbedding(embedding_dim=self.embedding_dim)
        self.jockey_embedder = JockeyEmbedding(embedding_dim=self.embedding_dim)
        self.course_embedder = CourseEmbedding(embedding_dim=10)  # Note: still hardcoded 10
        self.couple_embedder = CoupleEmbedding(embedding_dim=self.embedding_dim)

        # Track whether embeddings have been fitted
        self.embeddings_fitted = False

        # Store preprocessing parameters
        self.preprocessing_params = {
            'sequence_length': self.sequence_length
        }

        self.target_info = {
            'column': 'final_position',
            'type': feature_config['default_task_type']
        }

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.feature_store_dir, exist_ok=True)
        
        # Automatically enable GPU for embedding computations if available
        self._auto_enable_gpu()

        if self.verbose:
            print(f"Orchestrator initialized with:")
            print(f"  - SQLite path: {self.sqlite_path}")
            print(f"  - Cache directory: {self.cache_dir}")
            print(f"  - Feature store directory: {self.feature_store_dir}")
            print(f"  - Embedding dimension: {self.embedding_dim}")
            print(f"  - Sequence length: {self.sequence_length}")
            print(f"  - Cache enabled: {self.use_cache}")

    def log_info(self, message):
        """Simple logging method for backward compatibility."""
        if hasattr(self, 'verbose') and self.verbose:
            print(message)

    def enable_gpu_for_prediction(self):
        """Enable GPU for prediction feature preparation (embedding calculations)."""
        gpu_enabled = []
        
        if self.horse_embedder.enable_gpu_for_prediction():
            gpu_enabled.append("HorseEmbedding")
        if self.couple_embedder.enable_gpu_for_prediction():
            gpu_enabled.append("CoupleEmbedding")
            
        if gpu_enabled:
            print(f"FeatureOrchestrator: GPU enabled for prediction feature preparation")
            print(f"  GPU-accelerated components: {', '.join(gpu_enabled)}")
            return True
        else:
            print(f"FeatureOrchestrator: No GPU acceleration available")
            return False

    def disable_gpu_for_prediction(self):
        """Disable GPU and return all embedding models to CPU."""
        self.horse_embedder.disable_gpu_for_prediction()
        self.couple_embedder.disable_gpu_for_prediction()
        print(f"FeatureOrchestrator: GPU disabled, all models on CPU")
    
    def _auto_enable_gpu(self):
        """Automatically enable GPU for embedding computations if available."""
        try:
            import torch
            
            # Check if GPU acceleration is available
            gpu_available = False
            gpu_type = "None"
            
            if torch.backends.mps.is_available():
                gpu_available = True
                gpu_type = "MPS (Apple Silicon)"
            elif torch.cuda.is_available():
                gpu_available = True  
                gpu_type = "CUDA"
            
            if gpu_available:
                # Enable GPU for embeddings
                if self.enable_gpu_for_prediction():
                    if self.verbose:
                        print(f"FeatureOrchestrator: Auto-enabled {gpu_type} for embedding computations")
                    return True
                else:
                    if self.verbose:
                        print(f"FeatureOrchestrator: {gpu_type} detected but failed to enable")
            else:
                if self.verbose:
                    print(f"FeatureOrchestrator: No GPU acceleration available, using CPU")
            
        except ImportError:
            if self.verbose:
                print(f"FeatureOrchestrator: PyTorch not available, using CPU for embeddings")
        except Exception as e:
            if self.verbose:
                print(f"FeatureOrchestrator: Failed to enable GPU: {e}")
        
        return False

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
                # Use a simple cache approach with the historical_data cache type
                cached_data = self.cache_manager.load_dataframe('historical_data')
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
                self.cache_manager.save_dataframe(expanded_df, 'historical_data')
            except Exception as e:
                print(f"Warning: Could not save to cache: {str(e)}")

        return expanded_df

    def _expand_participants(self, df_races):
        """
        Clean version: expand participants and convert race results to final_position.
        """
        race_dfs = []

        for _, race in df_races.iterrows():
            try:
                # Skip if no participants
                if pd.isna(race['participants']):
                    continue

                participants = json.loads(race['participants'])
                if not participants:
                    continue

                # Create participant DataFrame
                race_df = pd.DataFrame(participants)

                # Add race metadata to each participant
                for col in df_races.columns:
                    if col not in ['participants', 'ordre_arrivee']:
                        race_df[col] = race[col]

                # Convert 'cl' to 'final_position'
                if 'cl' in race_df.columns:
                    race_df['final_position'] = self.convert_race_results_to_numeric(race_df['cl'], drop_empty=False)
                    race_df = race_df.drop(columns=['cl'])

                race_dfs.append(race_df)

            except (json.JSONDecodeError, KeyError):
                continue

        # Combine all races
        if race_dfs:
            return pd.concat(race_dfs, ignore_index=True)
        else:
            return pd.DataFrame()


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
#            'race_count': df['comp'].nunique(),
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
                # FIX: Convert dictionary to DataFrame before saving to cache
                embeddings_status = pd.DataFrame({'embeddings_fitted': [True]})
                # FIX: Pass the cache_key directly as the cache_type
                self.cache_manager.save_dataframe(embeddings_status, cache_key)
            except Exception as e:
                if self.verbose:
                    print(f"Cache save failed: {e}")

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


        # Cache the result
        if use_cache:
            try:
                # FIX: Pass cache_key directly as the cache_type, remove 'features' parameter
                self.cache_manager.save_dataframe(processed_df, cache_key)
            except Exception as e:
                print(f"Warning: Could not cache prepared features: {str(e)}")

        return processed_df
        
    def apply_embeddings(self, df, use_cache=True):
        """
        Apply fitted embeddings to the data.
        
        Args:
            df: DataFrame with prepared features
            use_cache: Whether to use caching
            
        Returns:
            DataFrame with embeddings applied
        """
        if not self.embeddings_fitted:
            raise ValueError("Embeddings must be fitted first. Call fit_embeddings() before applying embeddings.")
        
        processed_df = df.copy()
        
        # Apply horse embeddings
        if 'idche' in processed_df.columns:
            try:
                if hasattr(self.horse_embedder, 'generate_embeddings'):
                    horse_emb_dict = self.horse_embedder.generate_embeddings(processed_df)
                    for horse_id, embedding in horse_emb_dict.items():
                        mask = processed_df['idche'] == horse_id
                        for j, emb_val in enumerate(embedding):
                            if j < self.embedding_dim:
                                processed_df.loc[mask, f'horse_emb_{j}'] = emb_val
                else:
                    # Initialize with zeros if no method available
                    for j in range(self.embedding_dim):
                        processed_df[f'horse_emb_{j}'] = 0.0
            except Exception as e:
                print(f"Warning: Could not apply horse embeddings: {e}")
                for j in range(self.embedding_dim):
                    processed_df[f'horse_emb_{j}'] = 0.0
                    
        # Apply jockey embeddings  
        if 'idJockey' in processed_df.columns:
            try:
                if hasattr(self.jockey_embedder, 'transform_batch'):
                    jockey_df = self.jockey_embedder.transform_batch(processed_df)
                    # Copy jockey embedding columns if they exist
                    for col in jockey_df.columns:
                        if col.startswith('jockey_emb_') and col in jockey_df.columns:
                            processed_df[col] = jockey_df[col]
                else:
                    # Initialize with zeros if no method available
                    for j in range(self.embedding_dim):
                        processed_df[f'jockey_emb_{j}'] = 0.0
            except Exception as e:
                print(f"Warning: Could not apply jockey embeddings: {e}")
                for j in range(self.embedding_dim):
                    processed_df[f'jockey_emb_{j}'] = 0.0
                    
        # Apply couple embeddings
        if 'idche' in processed_df.columns and 'idJockey' in processed_df.columns:
            try:
                if hasattr(self.couple_embedder, 'generate_embeddings'):
                    couple_emb_dict = self.couple_embedder.generate_embeddings(processed_df)
                    for couple_id, embedding in couple_emb_dict.items():
                        # Parse couple_id which should be "horse_id_jockey_id"
                        horse_id, jockey_id = couple_id.split('_')
                        mask = (processed_df['idche'] == int(horse_id)) & (processed_df['idJockey'] == int(jockey_id))
                        for j, emb_val in enumerate(embedding):
                            if j < self.embedding_dim:
                                processed_df.loc[mask, f'couple_emb_{j}'] = emb_val
                else:
                    # Initialize with zeros if no method available
                    for j in range(self.embedding_dim):
                        processed_df[f'couple_emb_{j}'] = 0.0
            except Exception as e:
                print(f"Warning: Could not apply couple embeddings: {e}")
                for j in range(self.embedding_dim):
                    processed_df[f'couple_emb_{j}'] = 0.0
                    
        # Apply course embeddings
        try:
            if hasattr(self.course_embedder, 'generate_embeddings'):
                course_emb_dict = self.course_embedder.generate_embeddings(processed_df)
                for course_id, embedding in course_emb_dict.items():
                    mask = processed_df['comp'] == course_id
                    for j, emb_val in enumerate(embedding):
                        if j < 10:  # Course uses 10 dims
                            processed_df.loc[mask, f'course_emb_{j}'] = emb_val
            else:
                # Initialize with zeros if no method available
                for j in range(10):
                    processed_df[f'course_emb_{j}'] = 0.0
        except Exception as e:
            print(f"Warning: Could not apply course embeddings: {e}")
            for j in range(10):
                processed_df[f'course_emb_{j}'] = 0.0
        
        return processed_df

    def _detect_target_column(self, df):
        """
        Auto-detect an appropriate target column from the DataFrame.

        Args:
            df: DataFrame to check

        Returns:
            Name of detected target column
        """
        # Priority order for target columns
        target_candidates = ['final_position', 'cl', 'narrivee', 'position']

        for candidate in target_candidates:
            if candidate in df.columns:
                self.log_info(f"Auto-detected target column: '{candidate}'")
                return candidate

        # If none of the priority candidates exist, look for anything with position in the name
        position_cols = [col for col in df.columns if 'position' in col.lower()]
        if position_cols:
            self.log_info(f"Using '{position_cols[0]}' as target column")
            return position_cols[0]

        # Show all columns to help with debugging
        self.log_info(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Could not detect an appropriate target column. Please specify target_column explicitly.")

    def prepare_target_variable(self, df, target_column=None, task_type=None):
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

        # First, convert the raw values to numeric using our specialized converter
        # This handles all the race-specific codes properly regardless of task type
        self.log_info(f"Converting target column '{target_column}' using specialized race result conversion")
        numeric_results = self.convert_race_results_to_numeric(df[target_column], drop_empty=False)

        # Convert to Series with original index
        numeric_positions = pd.Series(numeric_results, index=df.index)
        # Now process according to task type
        if task_type == 'regression':
            # For regression, we can directly use the numeric positions
            return numeric_positions

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
            numeric_results, valid_mask = self.convert_race_results_to_numeric(df[target_column])
            y = pd.Series(numeric_results, index=df.index)

            # Group by race ID to get the race context
            race_groups = df.groupby('comp')

            # Prepare race context dataframe with proper ranking
            race_contexts = []

            for comp, group in race_groups:
                # Rank horses within the race (lower position is better)
                group = group.copy()
                group['rank'] = y[group.index].rank(method='min', na_option='bottom')
                race_contexts.append(group)

            # Combine all groups back
            ranked_df = pd.concat(race_contexts)

            # Return both the ranking and the grouped dataframe for specialized ranking models
            return ranked_df['rank'], ranked_df

        else:
            raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression', 'classification', or 'ranking'")

        return y

    def convert_race_results_to_numeric(self, results_array, drop_empty=False):
        """
        Convert race result values to numeric, preserving meaning of non-numeric codes.

        Args:
            results_array: Array or Series of race results (mix of numeric positions and status codes)
            drop_empty: Whether to drop empty strings or convert them to a numeric value

        Returns:
            if drop_empty=True: tuple of (numeric_array, valid_mask)
            if drop_empty=False: numeric array with all values converted
        """
        # First convert to pandas Series for easier handling
        results = pd.Series(results_array)


        # Get current max numeric value to use as base for non-finishers
        try:
            numeric_results = pd.to_numeric(results, errors='coerce')
            max_position = numeric_results.max()
            # Use a safe default if max is NaN
            max_position = 20 if pd.isna(max_position) else max_position
        except:
            max_position = 20  # Default if we can't determine max

        # Create a dictionary for mapping special codes
        special_codes = {
            # Empty values (if not dropping them)
            '': max_position + 50,

            # Disqualifications (least bad of non-finishers)
            'D': max_position + 10,
            'DI': max_position + 10,
            'DP': max_position + 10,
            'DAI': max_position + 10,
            'DIS': max_position + 10,

            # Retired/Fell (medium bad)
            'RET': max_position + 20,
            'TOM': max_position + 20,
            'ARR': max_position + 20,
            'FER': max_position + 20,

            # Never started (worst outcome)
            'NP': max_position + 30,
            'ABS': max_position + 30
        }

        # Apply conversions
        def convert_value(val):
            # Check for empty strings
            if isinstance(val, str) and val.strip() == '':
                return np.nan if drop_empty else special_codes['']

            # If it's already numeric, return as is
            try:
                return float(val)
            except (ValueError, TypeError):
                # If it's a recognized code, map it
                if isinstance(val, str):
                    for code, value in special_codes.items():
                        if code in val.upper():  # Case insensitive matching
                            return value
                # Default for any other unrecognized string
                return max_position + 40

        # Apply conversion to each value
        numeric_results = np.array([convert_value(val) for val in results])

        if drop_empty:
            # Create mask of valid (non-empty) entries
            valid_mask = ~np.isnan(numeric_results)
            return numeric_results, valid_mask
        else:
            # Return just the numeric results
            return numeric_results
    def drop_embedded_raw_features(self, df, keep_identifiers=False, clean_features=True):
        """
        Drop raw features that have already been converted to embeddings and optionally
        clean remaining features for model training.

        Args:
            df: DataFrame with feature columns
            keep_identifiers: Whether to keep ID columns needed for further processing
            clean_features: Whether to clean remaining features (convert to numeric, handle NAs)

        Returns:
            DataFrame with processed features ready for modeling
        """
        # Make a copy to avoid modifying the input
        clean_df = df.copy()

        # Track modifications for logging
        modifications = {
            'dropped_embedded': [],
            'encoded': [],
            'converted_to_numeric': [],
            'dropped_problematic': [],
            'filled_na': []
        }

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
                'pourcVictChevalHippo', 'pourcPlaceChevalHippo', 'gainsAnneeEnCours',
                # Derived ratios
                'ratio_victoires', 'ratio_places', 'gains_par_course', 'perf_cheval_hippo',
                # Musique-derived features
                'musiqueche', 'age'  # age is typically included in horse embeddings
            ]

            # Add all che_* columns (global, weighted, bytype)
            che_columns = [col for col in clean_df.columns if
                           col.startswith(('che_global_', 'che_weighted_', 'che_bytype_'))]
            horse_raw_features.extend(che_columns)

            # Keep track of what's being dropped
            existing = [col for col in horse_raw_features if col in clean_df.columns]
            columns_to_drop.extend(existing)
            modifications['dropped_embedded'].extend(existing)

        # Jockey-related raw features
        if has_jockey_embeddings:
            jockey_raw_features = [
                'pourcVictJockHippo', 'pourcPlaceJockHippo', 'perf_jockey_hippo',
                'musiquejoc'
            ]

            # Add all joc_* columns
            joc_columns = [col for col in clean_df.columns if
                           col.startswith(('joc_global_', 'joc_weighted_', 'joc_bytype_'))]
            jockey_raw_features.extend(joc_columns)

            existing = [col for col in jockey_raw_features if col in clean_df.columns]
            columns_to_drop.extend(existing)
            modifications['dropped_embedded'].extend(existing)

        # Couple-related raw features
        if has_couple_embeddings:
            couple_raw_features = [
                'nbVictCouple', 'nbPlaceCouple', 'nbCourseCouple', 'TxVictCouple',
                'efficacite_couple', 'regularite_couple', 'progression_couple'
            ]
            existing = [col for col in couple_raw_features if col in clean_df.columns]
            columns_to_drop.extend(existing)
            modifications['dropped_embedded'].extend(existing)

        # Course-related raw features
        if has_course_embeddings:
            # These features are likely captured in course embeddings
            course_raw_features = [
                'hippo', 'dist', 'typec', 'meteo', 'temperature',
                'forceVent', 'directionVent', 'natpis', 'pistegp'
            ]
            existing = [col for col in course_raw_features if col in clean_df.columns]
            columns_to_drop.extend(existing)
            modifications['dropped_embedded'].extend(existing)

        # 3. Handle identifiers
        identifiers = ['idche', 'idJockey', 'idEntraineur', 'proprietaire', 'comp', 'couple_id']

        if not keep_identifiers:
            existing = [col for col in identifiers if col in clean_df.columns]
            columns_to_drop.extend(existing)

        # 4. Add other metadata columns that should generally be excluded from training
        metadata_columns = [
            'cheval', 'ordre_arrivee', 'participants', 'created_at', 'jour',
            'prix', 'reun', 'quinte', 'narrivee'
        ]
        existing = [col for col in metadata_columns if col in clean_df.columns]
        columns_to_drop.extend(existing)

        # 5. Only drop columns that actually exist in the dataframe
        columns_to_drop = [col for col in columns_to_drop if col in clean_df.columns]

        # 6. Drop the columns and save the list for reference
        clean_df = clean_df.drop(columns=columns_to_drop)

        # Save the list of dropped columns for reference
        self.preprocessing_params['dropped_raw_features'] = columns_to_drop

        # 7. Clean remaining features if requested
        if clean_features:
            # Process each remaining column
            for col in list(clean_df.columns):  # Use list to avoid modification during iteration
                # Skip embedding columns - they're already numeric
                if any(col.startswith(prefix) for prefix in
                       ['horse_emb_', 'jockey_emb_', 'couple_emb_', 'course_emb_']):
                    continue

                # Handle empty strings
                if clean_df[col].dtype == 'object':
                    empty_count = (clean_df[col] == '').sum()
                    if empty_count > 0:
                        clean_df[col] = clean_df[col].replace('', np.nan).infer_objects(copy=False)
                        modifications['filled_na'].append(f"{col} ({empty_count} empty strings)")

                # Convert to numeric if possible
                if not pd.api.types.is_numeric_dtype(clean_df[col]):
                    try:
                        # Try converting to numeric
                        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

                        # Check if conversion produced too many NAs - but preserve essential columns
                        na_pct = clean_df[col].isna().mean()
                        # NEVER drop essential race context columns during prediction
                        essential_columns = ['typec', 'natpis', 'meteo', 'hippo', 'dist', 'temperature', 'final_position']
                        
                        if col in essential_columns:
                            # For essential columns, fill with defaults instead of dropping
                            if col == 'typec':
                                clean_df[col] = clean_df[col].fillna('P')  # Default to Plat
                            elif col == 'natpis':
                                clean_df[col] = clean_df[col].fillna('PSF')  # Default surface
                            elif col == 'meteo':
                                clean_df[col] = clean_df[col].fillna('BEAU')  # Default weather
                            elif col == 'hippo':
                                clean_df[col] = clean_df[col].fillna('UNKNOWN')  # Default track
                            elif col in ['dist', 'temperature']:
                                median_val = clean_df[col].median()
                                if pd.isna(median_val):
                                    median_val = 1600 if col == 'dist' else 15  # Sensible defaults
                                clean_df[col] = clean_df[col].fillna(median_val)
                            elif col == 'final_position':
                                # Keep as NaN for prediction target - don't convert to numeric
                                continue
                            # Convert to categorical/numeric after filling
                            if col in ['typec', 'natpis', 'meteo', 'hippo']:
                                # Keep as categorical - create label encoding
                                unique_values = sorted(clean_df[col].unique())
                                categorical_col = pd.Categorical(clean_df[col], categories=unique_values)
                                clean_df[f"{col}_code"] = categorical_col.codes
                                modifications['encoded'].append(f"{col} (essential, label encoded)")
                            else:
                                modifications['converted_to_numeric'].append(f"{col} (essential, preserved)")
                        elif na_pct > 0.7:  # Only drop if >70% NAs (more conservative than 30%)
                            # For high-cardinality categorical, use label encoding
                            if col in ['corde', 'reunion', 'course', 'partant']:
                                # These are likely categorical - create a code version with safe encoding
                                orig_col = clean_df[col].copy()
                                # Fill NaN values first to avoid category issues
                                orig_col = orig_col.fillna('unknown')
                                # Create categorical with all unique values to avoid unseen category errors
                                unique_values = sorted(orig_col.unique())
                                if 'unknown' not in unique_values:
                                    unique_values.append('unknown')
                                categorical_col = pd.Categorical(orig_col, categories=unique_values)
                                clean_df[f"{col}_code"] = categorical_col.codes
                                clean_df = clean_df.drop(columns=[col])
                                modifications['encoded'].append(f"{col} (label encoding)")
                            else:
                                # Otherwise drop
                                clean_df = clean_df.drop(columns=[col])
                                modifications['dropped_problematic'].append(f"{col} (too many NAs: {na_pct:.1%})")
                        else:
                            # Fill remaining NAs with median
                            median_val = clean_df[col].median()
                            clean_df[col] = clean_df[col].fillna(median_val)
                            modifications['converted_to_numeric'].append(col)
                            if na_pct > 0:
                                modifications['filled_na'].append(
                                    f"{col} ({na_pct:.1%} NAs, filled with {median_val:.2f})")
                    except:
                        # If conversion fails completely, drop the column
                        clean_df = clean_df.drop(columns=[col])
                        modifications['dropped_problematic'].append(f"{col} (cannot convert to numeric)")

                # Fill NAs in numeric columns
                elif clean_df[col].isna().any():
                    median_val = clean_df[col].median()
                    na_count = clean_df[col].isna().sum()
                    clean_df[col] = clean_df[col].fillna(median_val)
                    modifications['filled_na'].append(f"{col} ({na_count} NAs, filled with {median_val:.2f})")

            # Final check for any remaining non-numeric columns
            non_numeric = [col for col in clean_df.columns
                           if not pd.api.types.is_numeric_dtype(clean_df[col])]
            if non_numeric:
                clean_df = clean_df.drop(columns=non_numeric)
                modifications['dropped_problematic'].extend(non_numeric)

        # Log summary of cleaning
        self.log_info("\n===== FEATURE PROCESSING SUMMARY =====")
        total_modifications = 0
        for category, items in modifications.items():
            if items:
                count = len(items)
                total_modifications += count
                self.log_info(f"\n{category.replace('_', ' ').title()} ({count}):")
                # Show first few items if there are many
                to_show = items if count <= 10 else items[:10] + [f"... and {count - 10} more"]
                for item in to_show:
                    self.log_info(f"  - {item}")

        initial_cols = len(df.columns)
        final_cols = len(clean_df.columns)
        self.log_info(f"\nTotal modifications: {total_modifications}")
        self.log_info(f"Final feature count: {final_cols} (reduced from {initial_cols})")

        return clean_df

    def extract_lstm_features(self, complete_df, sequence_length=None):
        """LSTM features removed - method kept for compatibility."""
        self.log_info("LSTM feature extraction removed from system")
        return None, None, None

    def prepare_tabnet_features(self, df, use_cache=True):
        """
        Prepare features specifically for TabNet model.
        TabNet can handle both numerical and categorical features directly.

        Args:
            df: DataFrame with race data
            use_cache: Whether to use caching

        Returns:
            DataFrame with TabNet-ready features
        """
                # Step 1: Preserve target column before any processing
        target_backup = None
        if 'final_position' in df.columns:
            target_backup = df['final_position'].copy()
            self.log_info("Preserving target column 'final_position' for TabNet")

        # Step 2: Apply basic feature preparation
        processed_df = self.prepare_features(df, use_cache=use_cache)

        # Step 3: TabNet-specific feature selection and preprocessing
        tabnet_features = []

        # Numerical features that TabNet can use directly
        numerical_cols = [
            'age', 'cotedirect', 'coteprob', 'dist', 'temperature', 'forceVent',
            'numero', 'pourcVictChevalHippo', 'pourcPlaceChevalHippo',
            'pourcVictJockHippo', 'pourcPlaceJockHippo', 'victoirescheval',
            'placescheval', 'coursescheval'
        ]

        # Add available numerical features
        for col in numerical_cols:
            if col in processed_df.columns:
                tabnet_features.append(col)

        # Categorical features (TabNet can handle these with label encoding)
        categorical_cols = ['natpis', 'typec', 'meteo', 'corde']
        # Label encode categorical features for TabNet
        from sklearn.preprocessing import LabelEncoder

        for col in categorical_cols:
            if col in processed_df.columns:
                le = LabelEncoder()
                # Handle missing values properly for categorical columns
                if pd.api.types.is_categorical_dtype(processed_df[col]):
                    # For categorical columns, add 'unknown' to categories first
                    if 'unknown' not in processed_df[col].cat.categories:
                        processed_df[col] = processed_df[col].cat.add_categories(['unknown'])
                    processed_df[col] = processed_df[col].fillna('unknown')
                else:
                    # For non-categorical columns, simple fillna works
                    processed_df[col] = processed_df[col].fillna('unknown')

                # Convert to string and encode
                processed_df[f'{col}_encoded'] = le.fit_transform(processed_df[col].astype(str))
                tabnet_features.append(f'{col}_encoded')

        # Step 4: Create TabNet feature dataframe
        tabnet_df = processed_df[tabnet_features].copy()

        # Step 5: Restore target column if it was backed up
        if target_backup is not None:
            tabnet_df['final_position'] = target_backup
            self.log_info("Restored target column 'final_position' after feature preparation")

        # Step 6: Fill any remaining NaN values with 0 (but preserve target column)
        feature_cols = [col for col in tabnet_df.columns if col != 'final_position']
        tabnet_df[feature_cols] = tabnet_df[feature_cols].fillna(0)

        return tabnet_df