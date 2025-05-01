import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LSTM model training")
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of races to process')
    parser.add_argument('--race-type', type=str, default=None, help='Filter by race type (e.g., "A" for Attele)')
    parser.add_argument('--date-filter', type=str, default=None, help='Filter by date (e.g., "jour > \'2023-01-01\'")')
    parser.add_argument('--db', type=str, default="dev", help='Database to use from config')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save feature store')
    parser.add_argument('--sequence-length', type=int, default=5, help='Length of sequences for LSTM')
    parser.add_argument('--step-size', type=int, default=1, help='Step size for sequence generation')
    parser.add_argument('--embedding-dim', type=int, default=16, help='Dimension for entity embeddings')
    parser.add_argument('--no-cache', action='store_true', help='Disable use of cache')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for feature store directory')
    args = parser.parse_args()

    # Initialize orchestrator with specified database
    config = AppConfig()
    sqlite_path = config.get_sqlite_dbpath(args.db)

    # Create orchestrator with specified settings
    orchestrator = FeatureEmbeddingOrchestrator(
        sqlite_path=sqlite_path,
        embedding_dim=args.embedding_dim
    )

    print(f"Preparing LSTM data with database: {args.db} ({sqlite_path})")

    # 1. Load data
    print("\n=== Step 1: Loading data ===")
    df = orchestrator.load_data(
        limit=args.limit,
        race_filter=args.race_type,
        date_filter=args.date_filter,
        use_cache=not args.no_cache
    )
    print(f"Loaded {len(df)} participant records from {df['comp'].nunique()} races")

    # 2. Preprocess data
    print("\n=== Step 2: Preprocessing data ===")
    processed_df = orchestrator.preprocess_data(df, use_cache=not args.no_cache)
    print(f"Preprocessed data shape: {processed_df.shape}")

    # 3. Apply embeddings
    print("\n=== Step 3: Applying embeddings ===")
    embedded_df = orchestrator.apply_embeddings(processed_df, use_cache=not args.no_cache, lstm_mode=True)
    print(f"Embedded data shape: {embedded_df.shape}")

    # 4. Prepare sequence data
    print("\n=== Step 4: Preparing sequence data ===")
    try:
        # Import StandardScaler here
        from sklearn.preprocessing import StandardScaler

        # Prepare sequences
        X_sequences, X_static, y = orchestrator.prepare_sequence_data(
            embedded_df,
            sequence_length=args.sequence_length,
            step_size=args.step_size
        )

        print(f"Created sequences:")
        print(f"  - X_sequences shape: {X_sequences.shape}")
        print(f"  - X_static shape: {X_static.shape}")
        print(f"  - y shape: {y.shape}")

        # 5. Split dataset for LSTM
        print("\n=== Step 5: Splitting dataset ===")
        from sklearn.model_selection import train_test_split

        # Split data (no need for group-based splitting since sequences are already prepared)
        X_seq_train, X_seq_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
            X_sequences, X_static, y, test_size=0.2, random_state=42
        )

        # Further split training data into train and validation
        X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
            X_seq_train, X_static_train, y_train, test_size=0.2, random_state=42
        )

        print(f"Split sizes:")
        print(f"  - Train: {len(X_seq_train)} sequences")
        print(f"  - Validation: {len(X_seq_val)} sequences")
        print(f"  - Test: {len(X_seq_test)} sequences")

        # 6. Save feature store for LSTM
        print("\n=== Step 6: Saving LSTM feature store ===")
        output_dir = args.output_dir or os.path.join(orchestrator.feature_store_dir, 'lstm')
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamp for directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_store_dir = os.path.join(output_dir, f"{args.prefix}lstm_feature_store_{timestamp}")
        os.makedirs(feature_store_dir, exist_ok=True)

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

        # Save metadata and README
        metadata = {
            'created_at': datetime.now().isoformat(),
            'sequence_length': args.sequence_length,
            'step_size': args.step_size,
            'embedding_dim': args.embedding_dim,
            'dataset_info': {
                'train_samples': len(X_seq_train),
                'val_samples': len(X_seq_val),
                'test_samples': len(X_seq_test),
                'sequence_shape': X_seq_train.shape,
                'static_shape': X_static_train.shape,
            },
            'feature_columns': orchestrator.preprocessing_params.get('feature_columns', []),
        }

        import json
        with open(os.path.join(feature_store_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save README
        with open(os.path.join(feature_store_dir, "README.md"), 'w') as f:
            f.write(f"# LSTM Feature Store {timestamp}\n\n")
            f.write("## Dataset Information\n")
            f.write(f"- Sequence length: {args.sequence_length}\n")
            f.write(f"- Step size: {args.step_size}\n")
            f.write(f"- Embedding dimension: {args.embedding_dim}\n")
            f.write(f"- Training samples: {len(X_seq_train)}\n")
            f.write(f"- Validation samples: {len(X_seq_val)}\n")
            f.write(f"- Test samples: {len(X_seq_test)}\n\n")

            f.write("## Files\n")
            f.write("- `X_seq_train.npy`, `X_seq_val.npy`, `X_seq_test.npy`: Sequence features\n")
            f.write("- `X_static_train.npy`, `X_static_val.npy`, `X_static_test.npy`: Static features\n")
            f.write("- `y_train.npy`, `y_val.npy`, `y_test.npy`: Target variables\n")
            f.write("- `metadata.json`: Dataset and preprocessing metadata\n\n")

            f.write("## Usage\n")
            f.write("To load this feature store:\n\n")
            f.write("```python\n")
            f.write("import numpy as np\n")
            f.write("import json\n\n")
            f.write("# Load features\n")
            f.write("X_seq_train = np.load('X_seq_train.npy')\n")
            f.write("X_static_train = np.load('X_static_train.npy')\n")
            f.write("y_train = np.load('y_train.npy')\n\n")
            f.write("# Load metadata\n")
            f.write("with open('metadata.json', 'r') as f:\n")
            f.write("    metadata = json.load(f)\n")
            f.write("```\n")

        print(f"LSTM feature store saved to {feature_store_dir}")

    except Exception as e:
        print(f"Error preparing sequence data: {str(e)}")
        import traceback
        traceback.print_exc()

        # Save what we have so far
        print("\n=== Saving regular feature store instead ===")
        X, y = orchestrator.prepare_training_dataset(embedded_df)
        X_train, X_val, X_test, y_train, y_val, y_test = orchestrator.split_dataset(X, y)

        feature_store_path = orchestrator.save_feature_store(
            X_train, X_val, X_test, y_train, y_val, y_test,
            prefix=f"{args.prefix}fallback_"
        )
        print(f"Regular feature store saved to {feature_store_path}")


if __name__ == '__main__':
    main()