from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, Any, Optional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor

# Import your new data handling modules here
from new_core.data_loader import load_race_data  # Update this import
from utils.env_setup import setup_environment, get_model_paths, get_cache_path
from model_training.features.features_engineering import EnhancedFeatureEngineering  # Your new feature engineering class
from new_core.models.architectures import create_hybrid_model
from new_core.utils.cache_manager import CacheManager


class HorseRaceModel:
    def __init__(self, config_path: str = 'config.yaml', model_name: str = 'hybrid',
                 sequence_length: int = 5):
        """Initialize the model with configuration."""
        self.config = setup_environment(config_path)
        self.model_paths = get_model_paths(self.config, model_name)
        self.sequence_length = sequence_length
        self.models: Optional[Dict[str, Any]] = None
        self.rf_model: Optional[RandomForestRegressor] = None
        self.lstm_model = None
        self.feature_engineering = EnhancedFeatureEngineering()  # Use your new feature engineering class
        self.history = None

        # Get active database type
        self.db_type = self.config['active_db']

        # Initialize cache manager with correct paths
        cache_dir = Path(get_cache_path(self.config, 'training_data', self.db_type)).parent
        self.cache_manager = CacheManager(cache_dir)
        print(f"Initialized with database type: {self.db_type}")
        print(f"Using cache directory: {cache_dir}")

        # New data handling configuration
        self.data_config = self.config.get('data_handling', {})
        print(f"Data handling configuration: {self.data_config}")

    def prepare_sequence_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training with new approach."""
        sequences = []
        static_features = []
        targets = []

        # Define sequential and static features based on new data structure
        sequential_features = self.data_config.get('sequential_features', [
            'position',
            'cotedirect',
            'dist',
            'musique_avg_position',
            'musique_top_3_rate'
        ])

        static_features_list = self.data_config.get('static_features', [
            'age',
            'temperature',
            'natpis',
            'typec',
            'meteo',
            'corde',
            'normalized_odds',
            'musique_fault_prone'
        ])

        print("\nFeature dimensions:")
        print(f"Sequential features: {len(sequential_features)}")
        print(f"Static features: {len(static_features_list)}")

        # Apply your new data preprocessing approach
        processed_df = self.feature_engineering.preprocess_features(df, sequential_features, static_features_list)

        # Group by horse using your new identifier field
        horse_id_field = self.data_config.get('horse_id_field', 'idche')
        date_field = self.data_config.get('date_field', 'jour')

        for horse_id in processed_df[horse_id_field].unique():
            horse_data = processed_df[processed_df[horse_id_field] == horse_id].sort_values(date_field)

            # Apply your new sequence construction logic
            if len(horse_data) >= self.sequence_length + 1:
                try:
                    # Get sequential features with new data handling
                    seq_features = horse_data[sequential_features].values.astype(np.float32)

                    # Get static features with new approach
                    static_feat = horse_data[static_features_list].iloc[-1].values.astype(np.float32)

                    # Apply any new data validation logic
                    if self.feature_engineering.validate_features(seq_features, static_feat):
                        # Create sequences with sliding window or your new approach
                        for i in range(len(horse_data) - self.sequence_length):
                            sequences.append(seq_features[i:i + self.sequence_length])
                            static_features.append(static_feat)

                            # Update target based on your new prediction goal
                            target_field = self.data_config.get('target_field', 'position')
                            target_value = float(seq_features[i + self.sequence_length,
                            sequential_features.index(target_field) if target_field in sequential_features else 0])
                            targets.append(target_value)
                except (ValueError, TypeError) as e:
                    print(f"Error processing horse {horse_id}: {e}")
                    continue

        if not sequences:
            raise ValueError("No valid sequences could be created. Check data quality.")

        # Convert to numpy arrays with explicit types
        sequences = np.array(sequences, dtype=np.float32)
        static_features = np.array(static_features, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        print("\nData shapes:")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Static features shape: {static_features.shape}")
        print(f"Targets shape: {targets.shape}")

        return sequences, static_features, targets

    def load_or_prepare_data(self, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load or prepare data for training models."""
        # Get cache paths
        historical_cache = get_cache_path(self.config, 'historical_data', self.db_type)
        training_cache = get_cache_path(self.config, 'training_data', self.db_type)

        # Apply your new caching strategy
        if use_cache and self.data_config.get('enable_caching', True):
            print(f"Attempting to load cached data from: {training_cache}")
            cached_data = self.cache_manager.load(training_cache)
            if cached_data is not None:
                print("Found cached training data")
                df_features = cached_data
            else:
                print("No cached data found, processing historical data...")
                # Apply your new data loading approach
                historical_data = self.cache_manager.load(historical_cache)
                if historical_data is None:
                    print("Loading raw historical data...")
                    # Use your new data loading function
                    historical_data = load_race_data(
                        db_type=self.db_type,
                        config=self.data_config
                    )
                    self.cache_manager.save(historical_data, historical_cache)

                print("Extracting features with new approach...")
                df_features = self.feature_engineering.extract_all_features(
                    historical_data,
                    config=self.data_config
                )
                self.cache_manager.save(df_features, training_cache)
        else:
            print("Cache disabled or new data handling requires fresh processing...")
            # Load data with your new approach
            df_historical = load_race_data(
                db_type=self.db_type,
                config=self.data_config
            )
            df_features = self.feature_engineering.extract_all_features(
                df_historical,
                config=self.data_config
            )

        # Prepare static features for RF with new approach
        print("\nPreparing features for training...")
        static_columns = self.feature_engineering.get_feature_columns(self.data_config)
        static_features_df = df_features[static_columns].astype(float)

        # Apply any new feature normalization or scaling
        if self.data_config.get('normalize_features', False):
            static_features_df = self.feature_engineering.normalize_features(static_features_df)

        return df_features, static_features_df

    def train_rf_model(self, static_features_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        """Train the Random Forest model separately."""
        print("\n===== TRAINING RANDOM FOREST MODEL =====")

        # Create or get RF model
        if self.models is None or 'rf' not in self.models:
            # Create a new RF model with custom parameters
            rf_params = self.data_config.get('rf_params', {})
            self.rf_model = RandomForestRegressor(
                n_estimators=rf_params.get('n_estimators', 100),
                max_depth=rf_params.get('max_depth', None),
                min_samples_split=rf_params.get('min_samples_split', 2),
                min_samples_leaf=rf_params.get('min_samples_leaf', 1),
                max_features=rf_params.get('max_features', 'auto'),
                n_jobs=rf_params.get('n_jobs', -1),
                random_state=rf_params.get('random_state', 42)
            )
        else:
            self.rf_model = self.models['rf']

        # Get target field
        target_field = self.data_config.get('target_field', 'position')

        # Train RF model
        print(f"Training RF model on {len(static_features_df)} samples...")
        print(f"Input shape: {static_features_df.shape}, Target: {target_field}")

        try:
            self.rf_model.fit(static_features_df, target_df[target_field])

            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': static_features_df.columns,
                'importance': self.rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop feature importance for RF model:")
            print(feature_importance.head(10))

            # Save feature importance
            if self.data_config.get('save_feature_importance', True):
                importance_path = Path(self.model_paths['logs']) / f'rf_feature_importance_{self.db_type}.csv'
                importance_path.parent.mkdir(parents=True, exist_ok=True)
                feature_importance.to_csv(importance_path, index=False)
                print(f"Saved feature importance to: {importance_path}")

            print("RF model training completed successfully!")

        except Exception as e:
            print(f"Error during RF model training: {str(e)}")
            raise

    def train_lstm_model(self, df_features: pd.DataFrame) -> None:
        """Train the LSTM model separately."""
        print("\n===== TRAINING LSTM MODEL =====")

        print("Preparing sequence data for LSTM...")
        seq_features, static_seq, targets = self.prepare_sequence_data(df_features)

        if len(seq_features) == 0:
            raise ValueError("No valid sequences found in the data")

        # Create LSTM model if needed
        if self.models is None or 'lstm' not in self.models:
            print("Creating new LSTM model architecture...")
            lstm_params = self.data_config.get('lstm_params', {})

            self.models = create_hybrid_model(
                sequence_length=self.sequence_length,
                seq_feature_dim=seq_features.shape[2],
                static_feature_dim=static_seq.shape[1],
                lstm_units=lstm_params.get('lstm_units', 64),
                dropout_rate=lstm_params.get('dropout_rate', 0.2)
            )

            self.lstm_model = self.models['lstm']
        else:
            self.lstm_model = self.models['lstm']

        # Configure callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.data_config.get('early_stopping_patience', 10),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.data_config.get('lr_reduction_factor', 0.5),
                patience=self.data_config.get('lr_patience', 5),
                min_lr=self.data_config.get('min_lr', 0.00001)
            )
        ]

        # Add model checkpoint
        if self.model_paths['logs']:
            checkpoint_path = Path(self.model_paths['logs']) / f'best_lstm_model_{self.db_type}.keras'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    str(checkpoint_path),
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                )
            )

        # Train LSTM model
        print(f"Training LSTM model on {len(seq_features)} sequences...")
        print(f"Sequence shape: {seq_features.shape}, Static shape: {static_seq.shape}")

        try:
            lstm_params = self.data_config.get('lstm_params', {})

            history = self.lstm_model.fit(
                [seq_features, static_seq],
                targets,
                epochs=lstm_params.get('epochs', 100),
                batch_size=lstm_params.get('batch_size', 32),
                validation_split=lstm_params.get('validation_split', 0.2),
                callbacks=callbacks,
                verbose=1
            )

            self.history = history

            # Save training history
            if self.data_config.get('save_training_history', True):
                history_path = Path(self.model_paths['logs']) / f'lstm_training_history_{self.db_type}.pkl'
                joblib.dump(history.history, history_path)
                print(f"Saved LSTM training history to: {history_path}")

            print("LSTM model training completed successfully!")

        except Exception as e:
            print(f"Error during LSTM model training: {str(e)}")
            print("\nDebugging information:")
            print(f"Sequence features shape: {seq_features.shape}")
            print(f"Static features shape: {static_seq.shape}")
            print(f"Targets shape: {targets.shape}")
            raise

    def train(self, use_cache: bool = True, train_rf: bool = True, train_lstm: bool = True) -> None:
        """Train either or both models based on parameters."""
        print(f"\nLoading historical race data for {self.db_type} database...")

        # Load or prepare data for training
        df_features, static_features_df = self.load_or_prepare_data(use_cache)

        # Create models dictionary if it doesn't exist
        if self.models is None:
            self.models = {}

        # Train RF model if requested
        if train_rf:
            self.train_rf_model(static_features_df, df_features)
            self.models['rf'] = self.rf_model

        # Train LSTM model if requested
        if train_lstm:
            self.train_lstm_model(df_features)
            self.models['lstm'] = self.lstm_model

        # Save models if any were trained
        if train_rf or train_lstm:
            self.save_models(save_rf=train_rf, save_lstm=train_lstm)

    def save_models(self, save_rf: bool = True, save_lstm: bool = True) -> None:
        """Save the trained models and components with new versioning approach."""
        # Apply your new versioning strategy
        version = self.data_config.get('model_version', '1.0.0')
        save_dir = Path(self.model_paths['model_path']) / f"v{version}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving models to: {save_dir} (version {version})")

        # Save RF model with new metadata
        if save_rf and self.rf_model is not None:
            rf_path = save_dir / self.model_paths['artifacts']['rf_model']

            # Add model metadata with new approach
            rf_metadata = {
                'model': self.rf_model,
                'version': version,
                'features': self.feature_engineering.get_feature_columns(self.data_config),
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            joblib.dump(rf_metadata, rf_path)
            print(f"Saved RF model with metadata to: {rf_path}")

        # Save LSTM model with new format
        if save_lstm and self.lstm_model is not None:
            lstm_path = save_dir / self.model_paths['artifacts']['lstm_model']
            self.lstm_model.save(lstm_path)
            print(f"Saved LSTM model to: {lstm_path}")

        # Save feature engineering state with new attributes
        feature_path = save_dir / self.model_paths['artifacts']['feature_engineer']
        feature_eng_state = {
            'feature_columns': self.feature_engineering.get_feature_columns(self.data_config),
            'feature_preprocessors': self.feature_engineering.get_preprocessors(),
            'feature_scalers': self.feature_engineering.get_scalers(),
            'position_history': self.feature_engineering.position_history,
            'jockey_stats': self.feature_engineering.jockey_stats,
            'trainer_stats': self.feature_engineering.get_trainer_stats(),  # New field
            'track_conditions': self.feature_engineering.get_track_conditions(),  # New field
            'n_jobs': self.feature_engineering.n_jobs
        }
        joblib.dump(feature_eng_state, feature_path)
        print(f"Saved enhanced feature engineering state to: {feature_path}")

        # Save model configuration
        config_path = save_dir / 'model_config.json'
        model_config = {
            'version': version,
            'sequence_length': self.sequence_length,
            'data_config': self.data_config,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_trained': {
                'rf': save_rf and self.rf_model is not None,
                'lstm': save_lstm and self.lstm_model is not None
            }
        }

        # Use your new config saving approach
        import json
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"Saved model configuration to: {config_path}")

        print(f"\nAll models and components saved successfully to {save_dir}")


if __name__ == "__main__":
    # Apply your new command line arguments approach
    import argparse

    parser = argparse.ArgumentParser(description='Train horse race prediction model with new approach')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of intermediate data')
    parser.add_argument('--sequence-length', type=int, default=5, help='Sequence length for LSTM')
    parser.add_argument('--model-name', type=str, default='hybrid', help='Model architecture name')
    parser.add_argument('--rf-only', action='store_true', help='Train only the Random Forest model')
    parser.add_argument('--lstm-only', action='store_true', help='Train only the LSTM model')

    args = parser.parse_args()

    # Determine which models to train
    train_rf = not args.lstm_only  # Train RF unless LSTM-only flag is set
    train_lstm = not args.rf_only  # Train LSTM unless RF-only flag is set

    trainer = NewHorseRaceModel(
        config_path=args.config,
        model_name=args.model_name,
        sequence_length=args.sequence_length
    )
    trainer.train(use_cache=not args.no_cache, train_rf=train_rf, train_lstm=train_lstm)