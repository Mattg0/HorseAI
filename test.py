# test.py
import os
import argparse
from utils.env_setup import AppConfig
from utils.cache_manager import CacheManager
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator


def test_cache_manager():
    """Test the CacheManager functionality."""
    from utils.cache_manager import CacheManager
    import pandas as pd

    # Create a test DataFrame
    test_df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': ['a', 'b', 'c']
    })

    # Initialize cache manager
    cache_manager = CacheManager()

    # Test cache type
    cache_type = "test_cache"

    # Clear any existing cache
    cache_manager.clear_cache(cache_type)

    # Test saving
    print(f"Saving DataFrame to cache type '{cache_type}'...")
    saved_df = cache_manager.save_dataframe(test_df, cache_type)
    print(f"Saved DataFrame")

    # Test loading
    print(f"Loading DataFrame from cache type '{cache_type}'...")
    loaded_df = cache_manager.load_dataframe(cache_type)

    if loaded_df is not None:
        print("Successfully loaded cached data:")
        print(loaded_df)
        return True
    else:
        print("Failed to load cached data")
        return False


def test_data_loading(orchestrator, args):
    """Test basic data loading functionality."""
    print("\n=== Testing data loading ===")
    # Checking for correct method name for loading data
    if hasattr(orchestrator, 'load_historical_races'):
        df = orchestrator.load_historical_races(limit=args.limit, race_filter=args.race_type)
    else:
        print("Error: 'load_historical_races' method not found")
        return None

    print(f"Successfully loaded {len(df)} participant records from {df['comp'].nunique()} races")
    print(f"Sample columns: {list(df.columns[:5])}")
    return df


def test_feature_preparation(orchestrator, df, args):
    """Test feature preparation and embedding."""
    print("\n=== Testing feature preparation ===")
    try:
        # Check for correct preprocessing method
        if hasattr(orchestrator, 'prepare_features'):
            print("Using prepare_features method...")
            processed_df = orchestrator.prepare_features(df, use_cache=not args.no_cache)
        elif hasattr(orchestrator, 'preprocess_data'):
            print("Using preprocess_data method...")
            processed_df = orchestrator.preprocess_data(df, use_cache=not args.no_cache)
        else:
            print("Error: No preprocessing method found")
            return None

        print(f"Successfully preprocessed data: {processed_df.shape}")

        # Apply embeddings
        if hasattr(orchestrator, 'apply_embeddings'):
            embedded_df = orchestrator.apply_embeddings(processed_df, use_cache=not args.no_cache)
            print(f"Successfully applied embeddings: {embedded_df.shape}")

            # Check for embedding columns
            embedding_cols = [col for col in embedded_df.columns if '_emb_' in col]
            print(f"Found {len(embedding_cols)} embedding columns")

            for prefix in ['horse', 'jockey', 'couple', 'course']:
                cols = [col for col in embedding_cols if col.startswith(f"{prefix}_emb_")]
                if cols:
                    print(f"  - {prefix.capitalize()} embeddings: {len(cols)}")

            return embedded_df
        else:
            print("No 'apply_embeddings' method found, using processed data")
            return processed_df  # Return processed data if embedding not available

    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_feature_store(orchestrator, embedded_df, args):
    """Test feature store saving and loading."""
    print("\n=== Testing feature store functionality ===")
    try:
        # Prepare training dataset
        print("Preparing training dataset...")
        if hasattr(orchestrator, 'prepare_training_dataset'):
            X, y = orchestrator.prepare_training_dataset(embedded_df)
        else:
            print("Error: No 'prepare_training_dataset' method found")
            return False

        print(f"Successfully prepared dataset with {X.shape[1]} features and {len(y)} samples")

        # Split dataset
        print("Splitting dataset...")
        if hasattr(orchestrator, 'split_dataset'):
            X_train, X_val, X_test, y_train, y_val, y_test = orchestrator.split_dataset(X, y)
        else:
            # Use scikit-learn directly if the method doesn't exist
            print("No 'split_dataset' method found, using scikit-learn directly")
            from sklearn.model_selection import train_test_split

            # First split: train+val vs test (0.2)
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Second split: train vs val (0.125 of original = 0.1/0.8)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=0.125, random_state=42
            )

        print(f"Successfully split dataset:")
        print(f"  - Training: {X_train.shape[0]} samples")
        print(f"  - Validation: {X_val.shape[0]} samples")
        print(f"  - Testing: {X_test.shape[0]} samples")

        # Save feature store
        print("\n=== Testing feature store saving ===")
        if hasattr(orchestrator, 'save_feature_store'):
            feature_store_path = orchestrator.save_feature_store(
                X_train, X_val, X_test, y_train, y_val, y_test,
                prefix=f"{args.db}_test_"
            )
        else:
            print("Error: No 'save_feature_store' method found")
            return False

        print(f"Successfully saved feature store to {feature_store_path}")

        # Load feature store
        print("\n=== Testing feature store loading ===")
        if hasattr(orchestrator, 'load_feature_store'):
            loaded_data = orchestrator.load_feature_store(feature_store_path=feature_store_path)
        else:
            print("Error: No 'load_feature_store' method found")
            return False

        if len(loaded_data) >= 6:  # Ensure we got at least 6 elements: X_train...y_test
            loaded_X_train, loaded_X_val, loaded_X_test, loaded_y_train, loaded_y_val, loaded_y_test = loaded_data[:6]
            metadata = loaded_data[6] if len(loaded_data) > 6 else {}

            print(f"Successfully loaded feature store from {feature_store_path}")
            print(f"Loaded data shapes match original: "
                  f"{loaded_X_train.shape == X_train.shape}, "
                  f"{loaded_y_train.shape == y_train.shape}")

            # Print some metadata
            if metadata:
                print("\nFeature store metadata:")
                if 'created_at' in metadata:
                    print(f"  - Created: {metadata['created_at']}")
                if 'dataset_info' in metadata:
                    info = metadata['dataset_info']
                    print(f"  - Training samples: {info.get('train_samples')}")
                    print(f"  - Feature count: {info.get('feature_count')}")

                    # Count embedding features
                    if 'embedding_features' in info:
                        embedding_types = {
                            'horse': [col for col in info['embedding_features'] if col.startswith('horse_emb_')],
                            'jockey': [col for col in info['embedding_features'] if col.startswith('jockey_emb_')],
                            'couple': [col for col in info['embedding_features'] if col.startswith('couple_emb_')],
                            'course': [col for col in info['embedding_features'] if col.startswith('course_emb_')]
                        }

                        print(f"  - Embedding features:")
                        for entity, cols in embedding_types.items():
                            if cols:
                                print(f"    - {entity.capitalize()}: {len(cols)} dimensions")

            return True
        else:
            print(f"Error: Unexpected format returned from load_feature_store")
            return False
    except Exception as e:
        print(f"Error in feature store testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_lstm_data_preparation(orchestrator, embedded_df, args):
    """Test LSTM sequence data preparation."""
    print("\n=== Testing LSTM data preparation ===")
    try:
        # Prepare sequence data
        sequence_length = 3  # Use a small value for testing
        print(f"Preparing sequence data with length={sequence_length}...")

        if not hasattr(orchestrator, 'prepare_sequence_data'):
            print("Error: No 'prepare_sequence_data' method found")
            return False

        try:
            X_sequences, X_static, y = orchestrator.prepare_sequence_data(
                embedded_df,
                sequence_length=sequence_length,
                step_size=1
            )

            print(f"Successfully prepared sequence data:")
            print(f"  - Sequences shape: {X_sequences.shape}")
            print(f"  - Static features shape: {X_static.shape}")
            print(f"  - Targets shape: {y.shape}")

            # Split the data (simple random split for testing)
            from sklearn.model_selection import train_test_split

            print("Splitting sequence data...")
            # Split data
            X_seq_train, X_seq_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
                X_sequences, X_static, y, test_size=0.2, random_state=42
            )

            # Further split training data into train and validation
            X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
                X_seq_train, X_static_train, y_train, test_size=0.2, random_state=42
            )

            print(f"Split sequence data:")
            print(f"  - Train: {X_seq_train.shape[0]} sequences")
            print(f"  - Validation: {X_seq_val.shape[0]} sequences")
            print(f"  - Test: {X_seq_test.shape[0]} sequences")

            # Save LSTM feature store
            print("\n=== Testing LSTM feature store saving ===")
            import numpy as np
            from datetime import datetime
            import json

            # Create output directory
            output_dir = os.path.join(orchestrator.feature_store_dir, 'lstm')
            os.makedirs(output_dir, exist_ok=True)

            # Create timestamp for directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_store_dir = os.path.join(output_dir, f"{args.db}_test_lstm_feature_store_{timestamp}")
            os.makedirs(feature_store_dir, exist_ok=True)

            print(f"Saving LSTM data to {feature_store_dir}...")

            # Save numpy arrays
            np.save(os.path.join(feature_store_dir, "X_seq_train.npy"), X_seq_train)
            np.save(os.path.join(feature_store_dir, "X_seq_val.npy"), X_seq_val)
            np.save(os.path.join(feature_store_dir, "X_seq_test.npy"), X_seq_test)

            np.save(os.path.join(feature_store_dir, "X_static_train.npy"), X_static_train)
            np.save(os.path.join(feature_store_dir, "X_static_val.npy"), X_static_val)
            np.save(os.path.join(feature_store_dir, "X_static_test.npy"), X_static_test)

            np.save(os.path.join(feature_store_dir, "y_train.npy"), y_train)
            np.save(os.path.join(feature_store_dir, "y_val.npy"), y_val)
            np.save(os.path.join(feature_store_dir, "y_test.npy"), y_test)

            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'sequence_length': sequence_length,
                'step_size': 1,
                'embedding_dim': orchestrator.embedding_dim,
                'dataset_info': {
                    'train_samples': len(X_seq_train),
                    'val_samples': len(X_seq_val),
                    'test_samples': len(X_seq_test),
                    'sequence_shape': list(X_seq_train.shape),
                    'static_shape': list(X_static_train.shape),
                },
                'feature_columns': orchestrator.preprocessing_params.get('feature_columns', []),
            }

            with open(os.path.join(feature_store_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            print(f"Successfully saved LSTM feature store to {feature_store_dir}")

            # Test loading LSTM feature store if the method exists
            if hasattr(orchestrator, 'load_lstm_feature_store'):
                print("\n=== Testing LSTM feature store loading ===")
                try:
                    result = orchestrator.load_lstm_feature_store(feature_store_path=feature_store_dir)

                    if len(result) >= 9:  # We expect at least 9 elements
                        loaded_X_seq_train = result[0]
                        loaded_X_static_train = result[3]
                        loaded_y_train = result[6]

                        print(f"Successfully loaded LSTM feature store")
                        print(f"Loaded data shapes match original:")
                        print(f"  - X_sequences: {loaded_X_seq_train.shape == X_seq_train.shape}")
                        print(f"  - X_static: {loaded_X_static_train.shape == X_static_train.shape}")
                        print(f"  - y: {loaded_y_train.shape == y_train.shape}")
                    else:
                        print(f"Unexpected format returned from load_lstm_feature_store")

                except Exception as e:
                    print(f"Error loading LSTM feature store: {str(e)}")

            print("\nLSTM feature store testing completed successfully!")
            return True
        except ValueError as ve:
            print(f"ValueError in sequence preparation: {str(ve)}")
            print("This might be normal if there aren't enough sequences for the given horses.")
            return False

    except Exception as e:
        print(f"Unexpected error in LSTM data preparation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline(orchestrator, args):
    """Test the full pipeline with the orchestrator."""
    print("\n=== Testing full pipeline ===")
    try:
        # Call the existing run_pipeline method if it exists
        if hasattr(orchestrator, 'run_pipeline'):
            print("Running full pipeline...")
            result = orchestrator.run_pipeline(
                limit=args.limit,
                race_filter=args.race_type,
                use_cache=not args.no_cache
            )

            if isinstance(result, tuple) and len(result) >= 6:  # Ensure we got at least 6 elements (X_train...y_test)
                X_train, X_val, X_test, y_train, y_val, y_test = result[:6]
                feature_store_path = result[6] if len(result) > 6 else None

                print(f"\nPipeline results:")
                print(f"  - Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
                print(f"  - Validation set: {X_val.shape[0]} samples")
                print(f"  - Test set: {X_test.shape[0]} samples")

                # Print some embedding statistics
                embedding_cols = {
                    'horse': [col for col in X_train.columns if col.startswith('horse_emb_')],
                    'jockey': [col for col in X_train.columns if col.startswith('jockey_emb_')],
                    'couple': [col for col in X_train.columns if col.startswith('couple_emb_')],
                    'course': [col for col in X_train.columns if col.startswith('course_emb_')]
                }

                print("\nEmbedding features:")
                for entity, cols in embedding_cols.items():
                    if cols:
                        print(f"  - {entity.capitalize()}: {len(cols)} dimensions")

                if feature_store_path:
                    print(f"\nFeature store saved to: {feature_store_path}")

                print("\nFull pipeline test completed successfully!")
                return True
            else:
                print(f"Unexpected format returned from run_pipeline")
                return False
        else:
            # If run_pipeline doesn't exist, run the steps manually
            print("run_pipeline method not found. Running steps manually...")

            # 1. Load data
            df = test_data_loading(orchestrator, args)
            if df is None:
                return False

            # 2. Preprocess data
            embedded_df = test_feature_preparation(orchestrator, df, args)
            if embedded_df is None:
                return False

            # 3. Save feature store
            feature_store_result = test_feature_store(orchestrator, embedded_df, args)

            print("\nFull pipeline test completed by running individual steps")
            return feature_store_result

    except Exception as e:
        print(f"Error in full pipeline test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test the FeatureEmbeddingOrchestrator")
    parser.add_argument('--limit', type=int, default=10, help='Limit the number of races to process')
    parser.add_argument('--race-type', type=str, default=None, help='Filter by race type (e.g., "A" for Attele)')
    parser.add_argument('--run-full', action='store_true', help='Run the full pipeline')
    parser.add_argument('--test-load', action='store_true', help='Test only feature store loading')
    parser.add_argument('--test-lstm', action='store_true', help='Test LSTM data preparation')
    parser.add_argument('--no-cache', action='store_true', help='Disable use of cache')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before running')
    parser.add_argument('--db', type=str, default="dev", help='Database to use from config')
    parser.add_argument('--embedding-dim', type=int, default=8, help='Embedding dimension to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Initialize orchestrator with specified database
    config = AppConfig()
    sqlite_path = config.get_sqlite_dbpath(args.db)

    # Create orchestrator with only arguments we know exist
    try:
        # Try with all parameters first
        orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=sqlite_path,
            embedding_dim=args.embedding_dim,
            verbose=args.verbose
        )
    except TypeError:
        # If verbose is not supported, try without it
        print("Note: 'verbose' parameter not supported, creating orchestrator without it.")
        orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=sqlite_path,
            embedding_dim=args.embedding_dim
        )

    print(f"Testing orchestrator with database: {args.db} ({sqlite_path})")
    print(f"Using embedding dimension: {args.embedding_dim}")

    # Clear cache if requested
    if args.clear_cache and hasattr(orchestrator, 'cache_manager') and hasattr(orchestrator.cache_manager,
                                                                               'clear_cache'):
        print("Clearing cache...")
        orchestrator.cache_manager.clear_cache()

    # Test only loading if requested
    if args.test_load:
        print("\n=== Testing feature store loading only ===")
        try:
            # Check for list_feature_stores method
            if hasattr(orchestrator, '_list_feature_stores'):
                feature_stores = orchestrator._list_feature_stores()
            else:
                # Try direct path if method doesn't exist
                feature_store_dir = orchestrator.feature_store_dir
                if os.path.exists(feature_store_dir):
                    feature_stores = [d for d in os.listdir(feature_store_dir)
                                      if os.path.isdir(os.path.join(feature_store_dir, d))
                                      and d.startswith('feature_store_')]
                    feature_stores.sort(reverse=True)  # Most recent first
                else:
                    feature_stores = []

            if not feature_stores:
                print("No feature stores found. Run the test first to create a feature store.")
                return

            # Load the most recent feature store
            feature_store_path = os.path.join(orchestrator.feature_store_dir, feature_stores[0])
            print(f"Loading most recent feature store: {feature_store_path}")

            if hasattr(orchestrator, 'load_feature_store'):
                result = orchestrator.load_feature_store(feature_store_path=feature_store_path)
                if len(result) >= 6:
                    X_train, X_val, X_test, y_train, y_val, y_test = result[:6]

                    print(f"Successfully loaded feature store")
                    print(f"Loaded data shapes:")
                    print(f"  - X_train: {X_train.shape}")
                    print(f"  - y_train: {y_train.shape}")

                    print("\nFeature store loading test completed successfully!")
                else:
                    print(f"Unexpected format returned from load_feature_store")
            else:
                print("Error: No 'load_feature_store' method found")
            return
        except Exception as e:
            print(f"Error loading feature store: {str(e)}")
            import traceback
            traceback.print_exc()
            return

    # If running full pipeline test
    if args.run_full:
        test_full_pipeline(orchestrator, args)
        return

    # Otherwise run step by step tests

    # Test data loading
    df = test_data_loading(orchestrator, args)
    if df is None:
        print("Data loading failed, stopping tests.")
        return

    # Test feature preparation
    embedded_df = test_feature_preparation(orchestrator, df, args)
    if embedded_df is None:
        print("Feature preparation failed, stopping tests.")
        return

    # Test feature store functionality
    feature_store_result = test_feature_store(orchestrator, embedded_df, args)

    # Test LSTM data preparation if requested
    if args.test_lstm:
        test_lstm_data_preparation(orchestrator, embedded_df, args)

    print("\nAll tests completed!")


if __name__ == '__main__':
    main()