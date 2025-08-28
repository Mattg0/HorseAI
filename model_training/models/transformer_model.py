import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Concatenate, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple

from .base_model import BaseModel


class TransformerModel(BaseModel):
    """
    Simplified Transformer model for horse racing prediction.
    
    This model uses self-attention to identify relevant historical performances
    and learns temporal patterns in horse race data. It includes position encoding
    and processes both sequential and static features.
    """
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the Transformer model.
        
        Args:
            config: Model configuration dictionary
            verbose: Whether to print verbose output
        """
        super().__init__(config, verbose)
        
        # Extract transformer-specific config
        self.num_attention_heads = config.get('num_attention_heads', 4)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.early_stopping_patience = config.get('early_stopping_patience', 20)
        self.max_sequence_length = config.get('max_sequence_length', 10)
        
        if self.verbose:
            print(f"TransformerModel initialized with:")
            print(f"  Attention heads: {self.num_attention_heads}")
            print(f"  Hidden dim: {self.hidden_dim}")
            print(f"  Num layers: {self.num_layers}")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  Max sequence length: {self.max_sequence_length}")
    
    def prepare_features(self, X_sequences: np.ndarray, X_static: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for transformer processing.
        
        Args:
            X_sequences: Sequential features (batch_size, sequence_length, features)
            X_static: Static features (batch_size, static_features)
            
        Returns:
            Prepared sequential and static features
        """
        batch_size, sequence_length, n_features = X_sequences.shape
        
        # Truncate or pad sequences to max_sequence_length
        if sequence_length > self.max_sequence_length:
            # Take most recent sequences
            X_sequences = X_sequences[:, -self.max_sequence_length:, :]
        elif sequence_length < self.max_sequence_length:
            # Pad with zeros at the beginning
            pad_length = self.max_sequence_length - sequence_length
            padding = np.zeros((batch_size, pad_length, n_features))
            X_sequences = np.concatenate([padding, X_sequences], axis=1)
        
        return X_sequences, X_static
    
    def positional_encoding(self, length: int, depth: int) -> tf.Tensor:
        """
        Generate positional encoding for transformer.
        
        Args:
            length: Sequence length
            depth: Embedding depth
            
        Returns:
            Positional encoding tensor
        """
        depth = depth // 2
        positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
        
        angle_rates = 1 / (10000 ** depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)
        
        pos_encoding = np.concatenate([
            np.sin(angle_rads),
            np.cos(angle_rads)
        ], axis=-1)
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def transformer_block(self, x: tf.Tensor, name_prefix: str) -> tf.Tensor:
        """
        Single transformer block with multi-head attention and feedforward.
        
        Args:
            x: Input tensor
            name_prefix: Prefix for layer names
            
        Returns:
            Processed tensor
        """
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.hidden_dim // self.num_attention_heads,
            name=f'{name_prefix}_attention'
        )(x, x)
        
        # Add & Norm
        x = Add(name=f'{name_prefix}_add1')([x, attention_output])
        x = LayerNormalization(name=f'{name_prefix}_norm1')(x)
        
        # Feedforward network
        ff_output = Dense(
            self.hidden_dim * 2,
            activation='relu',
            name=f'{name_prefix}_ff1'
        )(x)
        ff_output = Dropout(self.dropout_rate, name=f'{name_prefix}_dropout1')(ff_output)
        ff_output = Dense(
            self.hidden_dim,
            name=f'{name_prefix}_ff2'
        )(ff_output)
        
        # Add & Norm
        x = Add(name=f'{name_prefix}_add2')([x, ff_output])
        x = LayerNormalization(name=f'{name_prefix}_norm2')(x)
        
        return x
    
    def build_model(self, input_shape: Tuple[int, ...]) -> Model:
        """
        Build the transformer model architecture.
        
        Args:
            input_shape: Tuple of (sequence_length, seq_features, static_features)
            
        Returns:
            Compiled Keras model
        """
        sequence_length, seq_features, static_features = input_shape
        
        # Sequential features input
        seq_input = Input(shape=(sequence_length, seq_features), name='sequential_features')
        
        # Static features input
        static_input = Input(shape=(static_features,), name='static_features')
        
        # Project sequential features to hidden dimension
        x = Dense(self.hidden_dim, name='input_projection')(seq_input)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(sequence_length, self.hidden_dim)
        x = x + pos_encoding
        
        # Apply transformer blocks
        for i in range(self.num_layers):
            x = self.transformer_block(x, f'transformer_block_{i}')
        
        # Global pooling to get sequence representation
        seq_representation = GlobalAveragePooling1D(name='global_pooling')(x)
        
        # Process static features
        static_hidden = Dense(
            self.hidden_dim // 2,
            activation='relu',
            name='static_processing'
        )(static_input)
        static_hidden = Dropout(self.dropout_rate, name='static_dropout')(static_hidden)
        
        # Combine sequential and static features
        combined = Concatenate(name='feature_fusion')([seq_representation, static_hidden])
        
        # Final prediction layers
        hidden = Dense(
            self.hidden_dim,
            activation='relu',
            name='final_hidden'
        )(combined)
        hidden = Dropout(self.dropout_rate, name='final_dropout')(hidden)
        
        # Output layer (regression for position prediction)
        output = Dense(1, activation='linear', name='position_output')(hidden)
        
        # Create model
        model = Model(
            inputs=[seq_input, static_input],
            outputs=output,
            name='TransformerHorseRacingModel'
        )
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        if self.verbose:
            print("Transformer model architecture:")
            model.summary()
        
        return model
    
    def train(self, X_sequences: np.ndarray, X_static: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the transformer model.
        
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
            print(f"Training transformer model with {len(y)} samples...")
        
        # Prepare features
        X_seq_prep, X_static_prep = self.prepare_features(X_sequences, X_static)
        
        # Build model if not already built
        if self.model is None:
            input_shape = (
                X_seq_prep.shape[1],  # Sequence length
                X_seq_prep.shape[2],  # Sequential features
                X_static_prep.shape[1]  # Static features
            )
            self.model = self.build_model(input_shape)
        
        # Split data for training and validation
        if validation_split > 0:
            (X_seq_train, X_seq_val,
             X_static_train, X_static_val,
             y_train, y_val) = train_test_split(
                X_seq_prep, X_static_prep, y,
                test_size=validation_split,
                random_state=42
            )
            
            validation_data = ([X_seq_val, X_static_val], y_val)
        else:
            X_seq_train, X_static_train, y_train = X_seq_prep, X_static_prep, y
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
                patience=7,
                min_lr=1e-6,
                verbose=1 if self.verbose else 0
            )
        ]
        
        # Train model
        history = self.model.fit(
            [X_seq_train, X_static_train],
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
            'model_type': 'transformer',
            'epochs_trained': len(history.history['loss']),
            'final_loss': float(final_loss),
            'final_mae': float(final_mae),
            'final_val_loss': float(final_val_loss) if final_val_loss else None,
            'final_val_mae': float(final_val_mae) if final_val_mae else None,
            'training_samples': len(X_seq_train),
            'validation_samples': len(X_seq_val) if validation_data else 0
        }
        
        if self.verbose:
            print(f"Transformer training completed:")
            print(f"  Epochs: {results['epochs_trained']}")
            print(f"  Final MAE: {results['final_mae']:.4f}")
            if final_val_mae:
                print(f"  Final Val MAE: {results['final_val_mae']:.4f}")
        
        return results
    
    def predict(self, X_sequences: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained transformer model.
        
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
        X_seq_prep, X_static_prep = self.prepare_features(X_sequences, X_static)
        
        # Make predictions
        predictions = self.model.predict(
            [X_seq_prep, X_static_prep],
            verbose=0
        )
        
        # Flatten predictions and ensure positive values
        predictions = predictions.flatten()
        predictions = np.maximum(predictions, 1.0)  # Minimum position is 1
        
        return predictions