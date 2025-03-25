import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple

# Import from existing code
from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator


class RacePredictor:
    """
    Race predictor that uses transformer models to predict race outcomes.
    Works with data converted by the RaceDataConverter.
    """

    def __init__(self, model_path: str, db_name: str = "dev"):
        """
        Initialize the race predictor.

        Args:
            model_path: Path to saved model directory
            db_name: Database configuration name from config
        """
        self.model_path = Path(model_path)

        # Check if model path exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path {self.model_path} does not exist")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize config
        self.config = AppConfig()

        # Get database path from config
        sqlite_path = self.config.get_sqlite_dbpath(db_name)

        # Initialize orchestrator for feature embedding
        self.orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=sqlite_path
        )

        # Load models and scalers
        self._load_models()

        print(f"Race predictor initialized with model at {model_path}")
        print(f"Using device: {self.device}")

    def _load_models(self):
        """Load the transformer and static models with their scalers."""
        # Load static model if available
        self.static_model = None
        static_model_path = self.model_path / "static_model.txt"
        if static_model_path.exists():
            try:
                import lightgbm as lgb
                self.static_model = lgb.Booster(model_file=str(static_model_path))
                print("Loaded static model")
            except ImportError:
                print("LightGBM not available, static model not loaded")

        # Load static scaler
        self.static_scaler = None
        static_scaler_path = self.model_path / "static_scaler.pt"
        if static_scaler_path.exists():
            self.static_scaler = torch.load(static_scaler_path, map_location=self.device)
            print("Loaded static scaler")

        # Load transformer model parameters
        self.transformer_params = None
        self.transformer_model = None
        transformer_params_path = self.model_path / "transformer_params.json"
        if transformer_params_path.exists():
            import json
            with open(transformer_params_path, 'r') as f:
                self.transformer_params = json.load(f)
            print("Loaded transformer parameters")

        # Load sequence scaler if available
        self.seq_scaler = None
        seq_scaler_path = self.model_path / "seq_scaler.pt"
        if seq_scaler_path.exists():
            self.seq_scaler = torch.load(seq_scaler_path, map_location=self.device)
            print("Loaded sequence scaler")

        # Load calibration model if available
        self.calibration_model = None
        calibration_model_path = self.model_path / "calibration_model.pkl"
        if calibration_model_path.exists():
            import pickle
            with open(calibration_model_path, 'rb') as f:
                self.calibration_model = pickle.load(f)
            print("Loaded calibration model")

    def _initialize_transformer(self, seq_feature_dim: int, static_feature_dim: int):
        """
        Initialize the transformer model with the correct dimensions.

        Args:
            seq_feature_dim: Sequence feature dimension
            static_feature_dim: Static feature dimension
        """
        # If we don't have transformer params or the model is already initialized, return
        if self.transformer_params is None or self.transformer_model is not None:
            return False

        # Import the transformer model class
        try:
            from race_prediction.predict_race import TransformerRaceModel
        except ImportError:
            # Define the class inline if import fails
            import torch.nn as nn

            class PositionalEncoding(nn.Module):
                """Positional encoding for transformer models."""

                def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
                    super(PositionalEncoding, self).__init__()
                    self.dropout = nn.Dropout(p=dropout)
                    pe = torch.zeros(max_len, d_model)
                    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    pe = pe.unsqueeze(0)
                    self.register_buffer('pe', pe)

                def forward(self, x):
                    x = x + self.pe[:, :x.size(1), :]
                    return self.dropout(x)

            class TransformerRaceModel(nn.Module):
                """Transformer-based model for sequence data in horse racing prediction."""

                def __init__(self, seq_feature_dim: int, static_feature_dim: int, num_heads: int = 4,
                             num_encoder_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.2):
                    super(TransformerRaceModel, self).__init__()
                    self.seq_feature_dim = seq_feature_dim
                    self.static_feature_dim = static_feature_dim
                    d_model = 64  # Transformer feature dimension

                    self.seq_projection = nn.Linear(seq_feature_dim, d_model)
                    self.static_projection = nn.Linear(static_feature_dim, d_model)
                    self.positional_encoding = PositionalEncoding(d_model, dropout)

                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
                        dropout=dropout, batch_first=True
                    )
                    self.transformer_encoder = nn.TransformerEncoder(
                        encoder_layer, num_layers=num_encoder_layers
                    )

                    self.sequence_feature_extractor = nn.Linear(d_model, d_model)
                    self.combined_layer = nn.Linear(d_model * 2, d_model)
                    self.output_layer = nn.Linear(d_model, 1)
                    self.relu = nn.ReLU()

                def forward(self, seq_features, static_features, src_mask=None):
                    batch_size, seq_len, _ = seq_features.shape
                    seq_projected = self.seq_projection(seq_features)
                    seq_encoded = self.positional_encoding(seq_projected)
                    transformer_output = self.transformer_encoder(seq_encoded, src_mask)
                    seq_representation = torch.mean(transformer_output, dim=1)
                    seq_representation = self.sequence_feature_extractor(seq_representation)
                    seq_representation = self.relu(seq_representation)
                    static_projected = self.static_projection(static_features)
                    static_projected = self.relu(static_projected)
                    combined = torch.cat([seq_representation, static_projected], dim=1)
                    combined = self.combined_layer(combined)
                    combined = self.relu(combined)
                    output = self.output_layer(combined)
                    return output.squeeze(-1)

        # Update dimensions in parameters
        self.transformer_params['seq_feature_dim'] = seq_feature_dim
        self.transformer_params['static_feature_dim'] = static_feature_dim

        # Create the model
        self.transformer_model = TransformerRaceModel(**self.transformer_params).to(self.device)

        # Load weights
        transformer_model_path = self.model_path / "transformer_model.pt"
        self.transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=self.device))
        self.transformer_model.eval()

        print(f"Initialized transformer model with seq_dim={seq_feature_dim}, static_dim={static_feature_dim}")
        return True

    def prepare_race_data(self, race_df: pd.DataFrame) -> Tuple[
        pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare race data for prediction by applying feature engineering and embeddings.

        Args:
            race_df: DataFrame with race and participant data

        Returns:
            Tuple of (static features, sequence features, static sequence features)
        """
        # Convert numeric fields if needed
        numeric_fields = [
            'age', 'cotedirect', 'coteprob', 'pourcVictChevalHippo',
            'pourcPlaceChevalHippo', 'pourcVictJockHippo', 'pourcPlaceJockHippo',
            'victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere',
            'gainsAnneeEnCours', 'nbCourseCouple', 'nbVictCouple', 'nbPlaceCouple',
            'TxVictCouple', 'recence', 'dist', 'temperature', 'forceVent',
            'ratio_victoires', 'ratio_places', 'gains_par_course',
            'efficacite_couple', 'regularite_couple', 'progression_couple',
            'perf_cheval_hippo', 'perf_jockey_hippo'
        ]

        for field in numeric_fields:
            if field in race_df.columns and not pd.api.types.is_numeric_dtype(race_df[field]):
                race_df[field] = pd.to_numeric(race_df[field], errors='coerce')

        # Apply embeddings
        embedded_df = self.orchestrator.apply_embeddings(race_df)

        # Extract static features
        X, _ = self.orchestrator.prepare_training_dataset(embedded_df)

        # Get sequence data if transformer model is configured
        X_seq = None
        X_static = None

        if self.transformer_params is not None and hasattr(self.orchestrator, 'prepare_sequence_data'):
            try:
                # Get sequence length from model info or use default
                sequence_length = self.transformer_params.get('sequence_length', 5)

                X_seq, X_static, _ = self.orchestrator.prepare_sequence_data(
                    embedded_df, sequence_length=sequence_length
                )

                # Initialize transformer if not already done
                if self.transformer_model is None and X_seq is not None and X_static is not None:
                    _, _, seq_feature_dim = X_seq.shape
                    static_feature_dim = X_static.shape[1]
                    self._initialize_transformer(seq_feature_dim, static_feature_dim)

            except Exception as e:
                print(f"Could not prepare sequence data: {str(e)}")
                X_seq = None
                X_static = None

        return X, X_seq, X_static

    def predict(self, X: pd.DataFrame, X_seq: Optional[np.ndarray] = None,
                X_static: Optional[np.ndarray] = None, blend_weight: float = 0.7) -> np.ndarray:
        """
        Make predictions using the model.

        Args:
            X: Static features DataFrame
            X_seq: Sequence features numpy array (optional)
            X_static: Static features for sequence model (optional)
            blend_weight: Weight for static model in blend (0-1)

        Returns:
            NumPy array with predictions
        """
        # Scale static features if scaler is available
        if self.static_scaler is not None:
            X_scaled = self.static_scaler.transform(X.values)
        else:
            X_scaled = X.values

        # Get static model predictions if available
        if self.static_model is not None:
            static_preds = self.static_model.predict(X_scaled)
        else:
            static_preds = None

        # Get transformer model predictions if available
        if self.transformer_model is not None and X_seq is not None and X_static is not None:
            # Scale sequence data if scaler is available
            if self.seq_scaler is not None:
                batch_size, seq_len, feat_dim = X_seq.shape
                X_seq_reshaped = X_seq.reshape(batch_size * seq_len, feat_dim)
                X_seq_scaled = self.seq_scaler.transform(X_seq_reshaped)
                X_seq_scaled = X_seq_scaled.reshape(batch_size, seq_len, feat_dim)
            else:
                X_seq_scaled = X_seq

            # Scale static data for transformer if scaler is available
            if self.static_scaler is not None:
                X_static_scaled = self.static_scaler.transform(X_static)
            else:
                X_static_scaled = X_static

            # Get predictions from transformer model
            self.transformer_model.eval()
            with torch.no_grad():
                X_seq_tensor = torch.tensor(X_seq_scaled, dtype=torch.float32).to(self.device)
                X_static_tensor = torch.tensor(X_static_scaled, dtype=torch.float32).to(self.device)
                transformer_preds = self.transformer_model(X_seq_tensor, X_static_tensor)
                transformer_preds = transformer_preds.cpu().numpy()

            # Blend predictions if both models are available
            if static_preds is not None:
                blended_preds = static_preds * blend_weight + transformer_preds * (1 - blend_weight)
            else:
                blended_preds = transformer_preds
        else:
            # Use static predictions if transformer isn't available
            blended_preds = static_preds if static_preds is not None else np.zeros(len(X))

        # Apply calibration if available
        if self.calibration_model is not None:
            calibrated_preds = self.calibration_model.calibrate(blended_preds)
            return calibrated_preds
        else:
            return blended_preds

    def predict_race(self, race_df: pd.DataFrame, blend_weight: float = 0.7) -> pd.DataFrame:
        """
        Predict race outcome.

        Args:
            race_df: DataFrame with race data (as produced by RaceDataConverter)
            blend_weight: Weight for static model in blend (0-1)

        Returns:
            DataFrame with predictions
        """
        # Save original DataFrame to add predictions to
        original_df = race_df.copy()

        # Prepare data for prediction
        X, X_seq, X_static = self.prepare_race_data(race_df)

        # Make predictions
        predictions = self.predict(X, X_seq, X_static, blend_weight)

        # Add predictions to original data
        result_df = original_df.copy()
        result_df['predicted_position'] = predictions

        # Sort by predicted position (ascending, better positions first)
        result_df = result_df.sort_values('predicted_position')

        # Add rank column
        result_df['predicted_rank'] = range(1, len(result_df) + 1)

        return result_df