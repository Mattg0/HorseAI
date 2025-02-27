# core/orchestrators/feature_embedding_orchestrator.py

import sqlite3
import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import argparse
import os

from core.calculators.static_feature_calculator import FeatureCalculator
from core.calculators.musique_calculation import MusiqueFeatureExtractor
from utils.env_setup import get_sqlite_dbpath


class FeatureEmbeddingOrchestrator:
    """
    Orchestrator for feature embedding and combination prior to model training.
    Handles loading historical race data, calculating features, and preparing datasets.
    """

    def __init__(self, sqlite_path=None):
        """
        Initialize the orchestrator.

        Args:
            sqlite_path: Path to SQLite database, if None uses default from env_setup
        """
        self.sqlite_path = sqlite_path or get_sqlite_dbpath()

    def load_historical_races(self, limit=None, filter_conditions=None, include_results=True):
        """
        Load historical race data from SQLite.

        Args:
            limit: Optional limit for number of races to load
            filter_conditions: Optional WHERE conditions for the query
            include_results: Whether to include race results

        Returns:
            DataFrame with historical race data and participants
        """
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

        if filter_conditions:
            query += f" WHERE {filter_conditions}"

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
                participants = json.loads(race['participants'])

                if participants:
                    # Create DataFrame for this race's participants
                    race_df = pd.DataFrame(participants)

                    # Add race information to each participant row
                    for col in df_races.columns:
                        if col != 'participants' and col != 'ordre_arrivee':
                            race_df[col] = race[col]

                    # Add result information if available
                    if 'ordre_arrivee' in race and race['ordre_arrivee'] and not pd.isna(race['ordre_arrivee']):
                        results = json.loads(race['ordre_arrivee'])
                        # Create a mapping of horse IDs to final positions
                        id_to_position = {res['idche']: res['narrivee'] for res in results}

                        # Add a column for the final position
                        race_df['final_position'] = race_df['idche'].map(id_to_position)

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

    def process_features(self, df):
        """
        Process and calculate all features for the dataset.

        Args:
            df: DataFrame with race and participant data

        Returns:
            DataFrame with all calculated features
        """
        # Make a copy to avoid modifying the original
        features_df = df.copy()

        # Apply static feature calculations
        features_df = FeatureCalculator.calculate_all_features(features_df)

        # Additional processing for specific features
        features_df = self._preprocess_features(features_df)

        return features_df

    def _preprocess_features(self, df):
        """
        Preprocess features: handle missing values, encode categorical features, etc.

        Args:
            df: DataFrame with calculated features

        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        processed_df = df.copy()

        # Handle missing numerical values
        numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(0)

        # Handle categorical features - encode as needed
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['comp', 'idche', 'cheval', 'final_position', 'jour']:
                # Convert to category type
                processed_df[col] = processed_df[col].astype('category')

        return processed_df

    def prepare_target_variable(self, df, target_column='final_position', task_type='regression'):
        """
        Prepare the target variable for training.

        Args:
            df: DataFrame with features and target
            target_column: Name of the target column
            task_type: 'regression' or 'classification'

        Returns:
            Processed target variable
        """
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

        else:
            raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression' or 'classification'")

        return y

    def prepare_training_dataset(self, df, target_column='final_position', task_type='regression'):
        """
        Prepare the final dataset for training.

        Args:
            df: DataFrame with calculated features
            target_column: Column to use as the target variable
            task_type: 'regression' or 'classification'

        Returns:
            X: Features DataFrame
            y: Target Series
        """
        # Drop rows with missing target values
        training_df = df.dropna(subset=[target_column])

        # Prepare the target variable
        y = self.prepare_target_variable(training_df, target_column, task_type)

        # Select feature columns (excluding target and non-feature columns)
        exclude_cols = [
            target_column, 'comp', 'idche', 'cheval', 'ordre_arrivee',
            'participants', 'created_at', 'jour', 'prix'
        ]

        # Remove columns that are in the exclude list and exist in the DataFrame
        X = training_df.drop(columns=[col for col in exclude_cols if col in training_df.columns])

        return X, y

    def split_dataset(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split the dataset into training, validation, and testing sets.

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        from sklearn.model_selection import train_test_split

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

    def run_complete_pipeline(self, limit=None, task_type='regression', test_size=0.2, val_size=0.1, random_state=42):
        """
        Run the complete pipeline from data loading to training set preparation.

        Args:
            limit: Optional limit for races to load
            task_type: 'regression' or 'classification'
            test_size: Proportion for test split
            val_size: Proportion for validation split
            random_state: Random seed

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Load data
        print("Loading historical race data...")
        df = self.load_historical_races(limit=limit)
        print(f"Loaded {len(df)} participant records from {df['comp'].nunique()} races")

        # Process features
        print("Processing features...")
        features_df = self.process_features(df)

        # Prepare training dataset
        print("Preparing training dataset...")
        X, y = self.prepare_training_dataset(features_df, task_type=task_type)
        print(f"Dataset prepared with {X.shape[1]} features and {len(y)} samples")

        # Split for training, validation, and testing
        print("Splitting dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(
            X, y, test_size=test_size, val_size=val_size, random_state=random_state
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_dataset(self, X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
        """
        Save the prepared datasets to disk.

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            y_train: Training targets
            y_val: Validation targets
            y_test: Test targets
            output_dir: Directory to save files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save features
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)

        # Save targets
        y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
        y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

        print(f"Datasets saved to {output_dir}")


def main():
    """
    Example usage of the feature embedding orchestrator.
    """
    parser = argparse.ArgumentParser(description='Run the feature embedding pipeline')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of races to load')
    parser.add_argument('--task', type=str, choices=['regression', 'classification'],
                        default='regression', help='Type of machine learning task')
    parser.add_argument('--output-dir', type=str, default='./data/processed',
                        help='Directory to save processed datasets')
    parser.add_argument('--save', action='store_true',
                        help='Save the processed datasets to disk')
    args = parser.parse_args()

    # Initialize the orchestrator
    orchestrator = FeatureEmbeddingOrchestrator()

    # Run the complete pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = orchestrator.run_complete_pipeline(
        limit=args.limit, task_type=args.task
    )

    # Print feature information
    print("\nTop 10 features:")
    for i, feature in enumerate(X_train.columns[:10]):
        print(f"  {i + 1}. {feature}")

    # Display target distribution
    if args.task == 'classification':
        print("\nTarget distribution:")
        print(y_train.value_counts(normalize=True))

    # Save datasets if requested
    if args.save:
        orchestrator.save_dataset(
            X_train, X_val, X_test, y_train, y_val, y_test, args.output_dir
        )


if __name__ == "__main__":
    main()