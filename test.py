import os
import argparse
from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Test the FeatureEmbeddingOrchestrator")
    parser.add_argument('--limit', type=int, default=10, help='Limit the number of races to process')
    parser.add_argument('--race-type', type=str, default=None, help='Filter by race type (e.g., "A" for Attele)')
    parser.add_argument('--run-full', action='store_true', help='Run the full pipeline')
    parser.add_argument('--save', action='store_true', help='Save feature store')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before running')
    parser.add_argument('--db', type=str, default="dev", help='Database to use from config')
    args = parser.parse_args()

    # Initialize orchestrator with specified database
    config = AppConfig()
    sqlite_path = config.get_sqlite_dbpath(args.db)
    orchestrator = FeatureEmbeddingOrchestrator(sqlite_path=sqlite_path)

    print(f"Testing orchestrator with database: {args.db} ({sqlite_path})")

    # Clear cache if requested
    if args.clear_cache:
        print("Clearing cache...")
        orchestrator.clear_cache()

    # Test data loading
    print("\n=== Testing data loading ===")

    df = orchestrator.load_historical_races(limit=args.limit, race_filter=args.race_type)
    print(f"Successfully loaded {len(df)} participant records from {df['comp'].nunique()} races")
    print(f"Sample columns: {df.columns[:5]}")


    if not args.run_full:
        print("\nTest completed. Use --run-full to test the complete pipeline.")
        return

    # Test feature preparation
    print("\n=== Testing feature preparation ===")
    try:
        features_df = orchestrator.prepare_features(df)
        print(f"Successfully prepared features with {features_df.shape[1]} columns")

        # Check for embedding columns
        embedding_cols = [col for col in features_df.columns if 'emb_' in col]
        print(f"Found {len(embedding_cols)} embedding columns")
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        return

    # Test dataset preparation
    print("\n=== Testing dataset preparation ===")
    try:
        X, y = orchestrator.prepare_training_dataset(features_df)
        print(f"Successfully prepared training dataset with {X.shape[1]} features and {len(y)} samples")
    except Exception as e:
        print(f"Error preparing training dataset: {str(e)}")
        return

    # Test dataset splitting
    print("\n=== Testing dataset splitting ===")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = orchestrator.split_dataset(X, y)
        print(f"Successfully split dataset:")
        print(f"  - Training: {X_train.shape[0]} samples")
        print(f"  - Validation: {X_val.shape[0]} samples")
        print(f"  - Testing: {X_test.shape[0]} samples")
    except Exception as e:
        print(f"Error splitting dataset: {str(e)}")
        return

    # Save feature store if requested
    if args.save:
        print("\n=== Saving feature store ===")
        try:
            feature_store_path = orchestrator.save_feature_store(
                X_train, X_val, X_test, y_train, y_val, y_test,
                prefix=f"{args.db}_test_"
            )
            print(f"Successfully saved feature store to {feature_store_path}")
        except Exception as e:
            print(f"Error saving feature store: {str(e)}")

    print("\nOrchestrator test completed successfully!")
    return orchestrator


if __name__ == '__main__':
    main()