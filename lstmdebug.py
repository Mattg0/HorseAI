#!/usr/bin/env python
# lstm_debug_with_real_data.py - Debug LSTM model with real race data

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import sqlite3
from typing import Optional, Tuple, Dict, List

# Add the project root to path to ensure imports work
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required project modules
from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from core.connectors.api_daily_sync import RaceFetcher

# ===== CONFIGURATION =====
MODEL_PATH = "models/2years/hybrid/2years_full_v20250409/hybrid_lstm_model.keras"
FEATURE_CONFIG_PATH = "models/2years/hybrid/2years_full_v20250409/hybrid_feature_engineer.joblib"
COMP_ID = "1576910"  # The specific race to analyze
DB_NAME = "2years"  # Database to use


def analyze_model(model_path):
    """Analyze a Keras model to identify input layer requirements"""
    print(f"\n===== ANALYZING MODEL: {model_path} =====")

    # Load the model
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
        model.summary()

        # Extract input shapes directly from model.inputs
        print("\n===== INPUT SHAPE ANALYSIS =====")
        for i, input_tensor in enumerate(model.inputs):
            # Fix: handle the shape attribute properly
            shape = input_tensor.shape
            # Convert to list if needed
            shape_list = [dim if dim is not None else None for dim in shape]

            print(f"Input {i + 1}: {input_tensor.name}")
            print(f"  Shape: {shape_list}")

            # For sequence inputs (3D)
            if len(shape) == 3:
                seq_length = shape[1]
                feature_count = shape[2]
                print(f"  Sequence Length: {seq_length}")
                print(f"  Feature Count: {feature_count}")

        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def load_feature_config(config_path):
    """Load and analyze feature engineering configuration"""
    print(f"\n===== LOADING FEATURE CONFIGURATION: {config_path} =====")

    if os.path.exists(config_path):
        try:
            feature_config = joblib.load(config_path)

            if isinstance(feature_config, dict):
                if 'sequence_length' in feature_config:
                    print(f"Sequence Length: {feature_config['sequence_length']}")
                if 'preprocessing_params' in feature_config:
                    params = feature_config['preprocessing_params']
                    if 'sequential_features' in params:
                        seq_features = params['sequential_features']
                        print(f"Sequential Features ({len(seq_features)}): {seq_features}")
                    if 'static_features' in params:
                        static_features = params['static_features']
                        print(f"Static Features ({len(static_features)}): {static_features}")
            else:
                print(f"Feature config type: {type(feature_config)}")
                if hasattr(feature_config, 'sequence_length'):
                    print(f"Sequence Length: {feature_config.sequence_length}")

            return feature_config
        except Exception as e:
            print(f"Error loading feature configuration: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Feature config not found at {config_path}")

    return None


def get_race_data(comp_id, db_name):
    """Get race data for a specific race"""
    print(f"\n===== FETCHING RACE DATA: comp={comp_id} =====")

    # Initialize the race fetcher
    config = AppConfig()
    db_path = config.get_sqlite_dbpath(db_name)
    race_fetcher = RaceFetcher(db_name=db_name, verbose=True)

    # Fetch the race
    race_data = race_fetcher.get_race_by_comp(comp_id)

    if race_data is None:
        print(f"Race {comp_id} not found in database")
        return None

    # Convert participants to DataFrame
    participants = race_data.get('participants', [])
    if isinstance(participants, str):
        try:
            participants = pd.read_json(participants)
        except:
            print("Error parsing participants JSON")
            participants = []

    if len(participants) == 0:
        print("No participants found for this race")
        return None

    # Create DataFrame
    df = pd.DataFrame(participants)

    # Add race information to DataFrame
    for field in ['typec', 'dist', 'natpis', 'meteo', 'temperature', 'jour',
                  'forceVent', 'directionVent', 'corde']:
        if field in race_data and race_data[field] is not None:
            df[field] = race_data[field]

    # Add comp to DataFrame
    df['comp'] = comp_id

    print(f"Loaded race data with {len(df)} participants")
    print(f"Columns: {df.columns.tolist()}")

    return df


def prepare_lstm_sequence_data(race_df, orchestrator, sequence_length=5):
    """Reproduce how LSTM sequence data is prepared during prediction"""
    print(f"\n===== PREPARING LSTM SEQUENCE DATA =====")

    # This replicates the logic in prepare_lstm_sequence_features but focuses on
    # how it would be used during prediction

    # First, apply embeddings to get enriched features
    try:
        embedded_df = orchestrator.apply_embeddings(
            race_df,
            clean_after_embedding=True,
            keep_identifiers=True,
            lstm_mode=True  # This ensures idche and jour are preserved
        )

        print(f"Embedded data shape: {embedded_df.shape}")
        print(f"Embedded columns: {embedded_df.columns.tolist()}")

        # Now try to fetch historical sequences
        X_seq, X_static, y, horse_ids, race_dates = orchestrator.prepare_lstm_sequence_features(
            embedded_df,
            sequence_length=sequence_length
        )

        print(f"Generated sequences with shape: {X_seq.shape}")
        print(f"Generated static features with shape: {X_static.shape}")
        print(f"Retrieved data for {len(horse_ids)} horses")

        # Check if we have sequences for all horses
        if len(horse_ids) < len(race_df):
            print(f"WARNING: Only generated sequences for {len(horse_ids)}/{len(race_df)} horses")
            print(f"Missing horses: {set(race_df['idche'].unique()) - set(horse_ids)}")

        return X_seq, X_static, horse_ids

    except Exception as e:
        print(f"Error preparing LSTM sequence data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def fetch_horse_sequences_debug(race_df, db_path, sequence_length=5):
    """Debug version of fetch_horse_sequences to diagnose sequence issues"""
    print(f"\n===== FETCHING HISTORICAL HORSE SEQUENCES =====")

    # Get all horse IDs from the race
    horse_ids = []
    if 'idche' in race_df.columns:
        # Filter out missing or invalid IDs
        horse_ids = [int(h) for h in race_df['idche'] if pd.notna(h)]

    if not horse_ids:
        print("No valid horse IDs found in race data")
        return None, None, None

    print(f"Found {len(horse_ids)} horses to fetch sequences for")

    # Connect to the database
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        # Define example sequential and static features
        # These should match what was used in training
        sequential_features = [
            'final_position', 'cotedirect', 'dist',
            # Include embeddings if available
            'horse_emb_0', 'horse_emb_1', 'horse_emb_2',
            'jockey_emb_0', 'jockey_emb_1', 'jockey_emb_2',
        ]

        static_features = [
            'age', 'temperature', 'natpis', 'typec', 'meteo', 'corde',
            'couple_emb_0', 'couple_emb_1', 'couple_emb_2',
        ]

        # Prepare containers for sequences
        all_sequences = []
        all_static_features = []
        all_horse_ids = []

        # For each horse, retrieve its historical races
        for horse_id in horse_ids:
            print(f"Processing historical data for horse {horse_id}")

            # Fetch historical races for this horse
            query = """
            SELECT hr.* 
            FROM historical_races hr
            WHERE hr.participants LIKE ?
            ORDER BY hr.jour DESC
            LIMIT 20
            """

            # Execute query
            cursor.execute(query, (f'%"idche": {horse_id}%',))
            horse_races = cursor.fetchall()

            if not horse_races:
                print(f"No historical races found for horse {horse_id}")
                continue

            print(f"Found {len(horse_races)} historical races for horse {horse_id}")

            # Show examples of what we found
            if len(horse_races) > 0:
                sample_race = horse_races[0]
                print(f"Sample race: {sample_race['comp']} on {sample_race['jour']}")

                # Print the participants structure
                try:
                    participants_sample = json.loads(sample_race['participants'])
                    horse_entry = next((p for p in participants_sample if int(p.get('idche', 0)) == horse_id), None)
                    if horse_entry:
                        print(f"Found horse in participants with keys: {horse_entry.keys()}")
                        # List a few important keys
                        for key in ['musiqueche', 'victoirescheval', 'placescheval']:
                            if key in horse_entry:
                                print(f"  {key}: {horse_entry[key]}")
                except Exception as e:
                    print(f"Error parsing participants: {str(e)}")

        conn.close()

        # If we couldn't generate sequences, show what else we could do
        if not all_sequences:
            print("\n===== FALLBACK OPTIONS =====")
            print("Since we couldn't retrieve proper historical sequences, we have these options:")
            print("1. Only use the RF model for predictions")
            print("2. Create synthetic sequences (less accurate)")
            print("3. Use a hybrid approach that weights RF more heavily when sequences are unavailable")

        return all_sequences, all_static_features, all_horse_ids

    except Exception as e:
        print(f"Error in fetch_horse_sequences_debug: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_prediction_with_model(model, X_seq, X_static):
    """Test model prediction with sequence and static data"""
    print(f"\n===== TESTING MODEL PREDICTION =====")

    if X_seq is None or X_static is None:
        print("Cannot test prediction: Missing input data")
        return

    try:
        # Check that shapes match model expectations
        expected_seq_shape = model.inputs[0].shape.as_list()
        expected_static_shape = model.inputs[1].shape.as_list()

        print(f"Model expects sequence shape: {expected_seq_shape}")
        print(f"Actual sequence data shape: {X_seq.shape}")

        print(f"Model expects static shape: {expected_static_shape}")
        print(f"Actual static data shape: {X_static.shape}")

        # Check if reshape is needed
        seq_needs_reshape = (len(X_seq.shape) != len(expected_seq_shape) or
                             X_seq.shape[1:] != tuple(d for d in expected_seq_shape[1:] if d is not None))

        static_needs_reshape = (len(X_static.shape) != len(expected_static_shape) or
                                X_static.shape[1:] != tuple(d for d in expected_static_shape[1:] if d is not None))

        if seq_needs_reshape:
            print(f"WARNING: Sequence data needs reshaping to match model expectations")
            # Try to reshape if possible (this is just for testing)
            if len(X_seq.shape) == 2 and len(expected_seq_shape) == 3:
                # If we have (batch, features) but need (batch, seq_len, features)
                # Assuming features can be reshaped into seq_len * new_features
                batch_size = X_seq.shape[0]
                if expected_seq_shape[1] is not None:
                    seq_len = expected_seq_shape[1]
                    # Calculate new feature dim to maintain the same total elements
                    total_elements = X_seq.shape[1]
                    if total_elements % seq_len == 0:
                        new_feat_dim = total_elements // seq_len
                        X_seq = X_seq.reshape(batch_size, seq_len, new_feat_dim)
                        print(f"Reshaped sequence data to: {X_seq.shape}")

        if static_needs_reshape:
            print(f"WARNING: Static data needs reshaping to match model expectations")
            # Could implement similar reshaping logic for static data if needed

        # Make prediction
        prediction = model.predict([X_seq, X_static], verbose=0)

        print(f"Prediction successful with shape: {prediction.shape}")
        print(f"Prediction values: {prediction.flatten()}")

    except Exception as e:
        print(f"Error testing prediction: {str(e)}")
        import traceback
        traceback.print_exc()





if __name__ == "__main__":
    # Initialize the configuration
    config = AppConfig()
    db_path = config.get_sqlite_dbpath(DB_NAME)

    # Initialize orchestrator
    orchestrator = FeatureEmbeddingOrchestrator(
        sqlite_path=db_path,
        verbose=True
    )

    # Load and analyze the model
    model = analyze_model(MODEL_PATH)

    # Load feature configuration
    feature_config = load_feature_config(FEATURE_CONFIG_PATH)

    # Update orchestrator with feature config if available
    if feature_config and isinstance(feature_config, dict):
        if 'preprocessing_params' in feature_config:
            orchestrator.preprocessing_params.update(feature_config['preprocessing_params'])
        if 'sequence_length' in feature_config:
            orchestrator.sequence_length = feature_config['sequence_length']

    # Get race data for specific race
    race_df = get_race_data(COMP_ID, DB_NAME)

    if race_df is not None:
        # Try preparing LSTM sequence data
        if model is not None:
            sequence_length = 5  # From model summary

            # Prepare sequences
            X_seq, X_static, horse_ids = prepare_lstm_sequence_data(
                race_df,
                orchestrator,
                sequence_length
            )

            # Debug direct sequence fetching
            fetch_horse_sequences_debug(race_df, db_path, sequence_length)

            # Test prediction with model if we have sequences
            if X_seq is not None and X_static is not None:
                test_prediction_with_model(model, X_seq, X_static)

