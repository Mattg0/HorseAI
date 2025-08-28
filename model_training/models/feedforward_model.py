import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple

from .base_model import BaseModel


class FeedforwardModel(BaseModel):
    """
    Feedforward Neural Network for horse racing prediction.
    
    This model processes sequential features by flattening them and using
    position embeddings to encode temporal information. It combines
    sequential and static features in a single feedforward network.
    """
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the Feedforward model.
        
        Args:
            config: Model configuration dictionary
            verbose: Whether to print verbose output
        """
        super().__init__(config, verbose)
        
        # Extract feedforward-specific config
        self.hidden_units = config.get('hidden_units', 96)
        self.learning_rate = config.get('learning_rate', 0.005)
        self.batch_size = config.get('batch_size', 64)
        self.epochs = config.get('epochs', 100)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.early_stopping_patience = config.get('early_stopping_patience', 15)
        self.position_encoding_dim = config.get('position_encoding_dim', 16)
        
        if self.verbose:
            print(f"FeedforwardModel initialized with:")
            print(f"  Hidden units: {self.hidden_units}")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Position encoding dim: {self.position_encoding_dim}")
    
    def prepare_features(self, X_sequences: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """
        Prepare features by flattening sequences and adding position embeddings.
        
        Args:
            X_sequences: Sequential features (batch_size, sequence_length, features)
            X_static: Static features (batch_size, static_features)
            
        Returns:
            Flattened feature array with position encoding
        """
        batch_size, sequence_length, n_features = X_sequences.shape
        
        # Flatten sequential features
        X_seq_flat = X_sequences.reshape(batch_size, -1)
        
        # Create position encoding for each timestep
        positions = np.arange(sequence_length).reshape(1, -1)
        positions = np.repeat(positions, batch_size, axis=0)
        
        # Apply recency weighting (more recent = higher weight)
        recency_weights = np.exp(np.arange(sequence_length) / sequence_length)
        recency_weights = recency_weights / recency_weights.sum()
        
        # Weight the sequential features by recency
        for i in range(sequence_length):
            start_idx = i * n_features
            end_idx = (i + 1) * n_features
            X_seq_flat[:, start_idx:end_idx] *= recency_weights[i]
        
        return X_seq_flat, X_static, positions
    
    def build_model(self, input_shape: Tuple[int, ...]) -> Model:
        """
        Build the feedforward model architecture.
        
        Args:
            input_shape: Tuple of (seq_features, static_features, sequence_length)
            
        Returns:
            Compiled Keras model
        """
        seq_features_size, static_features_size, sequence_length = input_shape
        
        # Sequential features input (flattened)
        seq_input = Input(shape=(seq_features_size,), name='sequential_features')
        
        # Static features input
        static_input = Input(shape=(static_features_size,), name='static_features')
        
        # Position embeddings input
        position_input = Input(shape=(sequence_length,), name='position_encoding')
        
        # Position embedding layer
        position_embedding = Embedding(
            input_dim=sequence_length,
            output_dim=self.position_encoding_dim,
            name='position_embedding'
        )(position_input)
        position_flat = Flatten()(position_embedding)
        
        # Combine all features
        combined = Concatenate(name='feature_fusion')([
            seq_input,
            static_input,
            position_flat
        ])
        
        # Hidden layer
        hidden = Dense(
            self.hidden_units,
            activation='relu',
            name='hidden_layer'
        )(combined)
        
        # Dropout for regularization
        hidden = Dropout(self.dropout_rate, name='dropout')(hidden)
        
        # Output layer (regression for position prediction)
        output = Dense(1, activation='linear', name='position_output')(hidden)
        
        # Create model
        model = Model(
            inputs=[seq_input, static_input, position_input],
            outputs=output,
            name='FeedforwardHorseRacingModel'
        )
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        if self.verbose:
            print("Feedforward model architecture:")
            model.summary()
        
        return model
    
    def train(self, X_sequences: np.ndarray, X_static: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the feedforward model.
        
        Args:
            X_sequences: Sequential training features
            X_static: Static training features
            y: Target values (horse positions)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results dictionary
        """
        # Assert data quality
        self.assert_data_quality(X_sequences, X_static, y)
        
        if self.verbose:
            print(f"Training feedforward model with {len(y)} samples...")
        
        # Prepare features
        X_seq_flat, X_static_prep, positions = self.prepare_features(X_sequences, X_static)
        
        # Build model if not already built
        if self.model is None:
            input_shape = (
                X_seq_flat.shape[1],  # Flattened sequential features
                X_static_prep.shape[1],  # Static features
                positions.shape[1]  # Sequence length for position encoding
            )
            self.model = self.build_model(input_shape)
        
        # Split data for training and validation
        if validation_split > 0:
            (X_seq_train, X_seq_val,
             X_static_train, X_static_val,
             pos_train, pos_val,
             y_train, y_val) = train_test_split(
                X_seq_flat, X_static_prep, positions, y,
                test_size=validation_split,
                random_state=42
            )
            
            validation_data = ([X_seq_val, X_static_val, pos_val], y_val)
        else:
            X_seq_train, X_static_train, pos_train, y_train = X_seq_flat, X_static_prep, positions, y
            validation_data = None
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1 if self.verbose else 0
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1 if self.verbose else 0
            )
        ]
        
        # Train model
        history = self.model.fit(
            [X_seq_train, X_static_train, pos_train],
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        self.is_trained = True
        self.training_history = {
            'loss': history.history['loss'],
            'mae': history.history['mae'],
            'epochs_trained': len(history.history['loss'])
        }
        
        if validation_data:
            self.training_history.update({
                'val_loss': history.history['val_loss'],
                'val_mae': history.history['val_mae']
            })
        
        # Calculate final training metrics
        final_loss = history.history['loss'][-1]
        final_mae = history.history['mae'][-1]
        
        if validation_data:
            final_val_loss = history.history['val_loss'][-1]
            final_val_mae = history.history['val_mae'][-1]
        else:
            final_val_loss = None
            final_val_mae = None
        
        # Assert learning progress
        initial_loss = history.history['loss'][0]
        assert final_loss < initial_loss, f"Model failed to learn: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}"
        
        results = {
            'status': 'success',
            'model_type': 'feedforward',
            'epochs_trained': len(history.history['loss']),
            'final_loss': float(final_loss),
            'final_mae': float(final_mae),
            'final_val_loss': float(final_val_loss) if final_val_loss else None,
            'final_val_mae': float(final_val_mae) if final_val_mae else None,
            'training_samples': len(X_seq_train),
            'validation_samples': len(X_seq_val) if validation_data else 0
        }
        
        if self.verbose:
            print(f"Feedforward training completed:")
            print(f"  Epochs: {results['epochs_trained']}")
            print(f"  Final MAE: {results['final_mae']:.4f}")
            if final_val_mae:
                print(f"  Final Val MAE: {results['final_val_mae']:.4f}")
        
        return results
    
    def predict(self, X_sequences: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained feedforward model.
        
        Args:
            X_sequences: Sequential features for prediction
            X_static: Static features for prediction
            
        Returns:
            Predicted horse positions
        """
        # Assert model is trained
        assert self.is_trained, "Model must be trained before making predictions"
        assert self.model is not None, "Model is not available"
        
        # Assert data quality (no y for prediction)
        self.assert_data_quality(X_sequences, X_static)
        
        # Prepare features
        X_seq_flat, X_static_prep, positions = self.prepare_features(X_sequences, X_static)
        
        # Make predictions
        predictions = self.model.predict(
            [X_seq_flat, X_static_prep, positions],
            verbose=0
        )
        
        # Flatten predictions and ensure positive values
        predictions = predictions.flatten()
        predictions = np.maximum(predictions, 1.0)  # Minimum position is 1
        
        return predictions