#!/usr/bin/env python3
"""
Prediction script for horse race model with calibration.
Takes a trained model and makes predictions on new race data.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import pickle
import torch
from typing import Dict, List, Union, Any, Optional, Tuple
from pathlib import Path

# Local imports
from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator

# Import the TransformerRaceModel class - this import pattern allows the script to work
# standalone while still having access to the model architecture
try:
    from train_race_model import TransformerRaceModel, PositionalEncoding, CalibrationModel
except ImportError:
    # Define the classes here as a fallback
    import torch.nn as nn
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression


    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer models."""

        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Create positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)

            # Register as buffer (not a parameter)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class TransformerRaceModel(nn.Module):
        """Transformer-based model for sequence data in horse racing prediction."""

        def __init__(self, seq_feature_dim: int, static_feature_dim: int, num_heads: int = 4,
                     num_encoder_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.2):
            super(TransformerRaceModel, self).__init__()

            # Define model dimensions
            self.seq_feature_dim = seq_feature_dim
            self.static_feature_dim = static_feature_dim
            d_model = 64  # Transformer feature dimension

            # Feature projection layers
            self.seq_projection = nn.Linear(seq_feature_dim, d_model)
            self.static_projection = nn.Linear(static_feature_dim, d_model)

            # Position encoding
            self.positional_encoding = PositionalEncoding(d_model, dropout)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers
            )

            # Output layers
            self.sequence_feature_extractor = nn.Linear(d_model, d_model)
            self.combined_layer = nn.Linear(d_model * 2, d_model)
            self.output_layer = nn.Linear(d_model, 1)

            # Activation functions
            self.relu = nn.ReLU()

        def forward(self, seq_features, static_features, src_mask=None):
            batch_size, seq_len, _ = seq_features.shape

            # Project sequence features to transformer dimension
            seq_projected = self.seq_projection(seq_features)

            # Add positional encoding
            seq_encoded = self.positional_encoding(seq_projected)

            # Pass through transformer encoder
            transformer_output = self.transformer_encoder(seq_encoded, src_mask)

            # Get global sequence representation (use the mean of all positions)
            seq_representation = torch.mean(transformer_output, dim=1)
            seq_representation = self.sequence_feature_extractor(seq_representation)
            seq_representation = self.relu(seq_representation)

            # Project static features
            static_projected = self.static_projection(static_features)
            static_projected = self.relu(static_projected)

            # Combine sequence and static features
            combined = torch.cat([seq_representation, static_projected], dim=1)
            combined = self.combined_layer(combined)
            combined = self.relu(combined)

            # Output prediction
            output = self.output_layer(combined)

            return output.squeeze(-1)


    class CalibrationModel:
        """
        Post-training calibration model to adjust race predictions.
        Helps correct systematic biases in model predictions.
        """

        def __init__(self, calibration_method: str = 'isotonic'):
            """
            Initialize a calibration model.

            Args:
                calibration_method: Method to use for calibration ('isotonic' or 'linear')
            """
            self.calibration_method = calibration_method
            self.model = None
            self.is_fitted = False

            # Statistics for analysis
            self.prediction_stats = None
            self.target_stats = None
            self.error_stats = None

            # Initialize the appropriate model
            if calibration_method == 'isotonic':
                self.model = IsotonicRegression(out_of_bounds='clip')
            elif calibration_method == 'linear':
                self.model = LinearRegression()
            else:
                raise ValueError(f"Unsupported calibration method: {calibration_method}")

        def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
            """
            Fit the calibration model based on predicted and true values.

            Args:
                y_pred: Predicted values
                y_true: True values
            """
            # Compute statistics
            self.prediction_stats = {
                'mean': np.mean(y_pred),
                'std': np.std(y_pred),
                'min': np.min(y_pred),
                'max': np.max(y_pred),
                'median': np.median(y_pred)
            }

            self.target_stats = {
                'mean': np.mean(y_true),
                'std': np.std(y_true),
                'min': np.min(y_true),
                'max': np.max(y_true),
                'median': np.median(y_true)
            }

            # Compute errors
            errors = y_pred - y_true
            self.error_stats = {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'min': np.min(errors),
                'max': np.max(errors),
                'median': np.median(errors),
                'mae': np.mean(np.abs(errors)),
                'rmse': np.sqrt(np.mean(np.square(errors)))
            }

            # Fit the model
            if self.calibration_method == 'isotonic':
                self.model.fit(y_pred, y_true)
            elif self.calibration_method == 'linear':
                self.model.fit(y_pred.reshape(-1, 1), y_true)

            self.is_fitted = True
            return self

        def calibrate(self, y_pred: np.ndarray) -> np.ndarray:
            """
            Calibrate predicted values.

            Args:
                y_pred: Predicted values to calibrate

            Returns:
                Calibrated predictions
            """
            if not self.is_fitted:
                raise ValueError("Calibration model has not been fitted yet")

            if self.calibration_method == 'isotonic':
                return self.model.transform(y_pred)
            elif self.calibration_method == 'linear':
                return self.model.predict(y_pred.reshape(-1, 1))

        def save(self, filepath: str):
            """
            Save the calibration model.

            Args:
                filepath: Path to save the model
            """
            if not self.is_fitted:
                raise ValueError("Cannot save: calibration model has not been fitted yet")

            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'calibration_method': self.calibration_method,
                    'prediction_stats': self.prediction_stats,
                    'target_stats': self.target_stats,
                    'error_stats': self.error_stats,
                    'is_fitted': self.is_fitted
                }, f)

        @classmethod
        def load(cls, filepath: str) -> 'CalibrationModel':
            """
            Load a saved calibration model.

            Args:
                filepath: Path to the saved model

            Returns:
                Loaded calibration model
            """
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            instance = cls(calibration_method=data['calibration_method'])
            instance.model = data['model']
            instance.is_fitted = data['is_fitted']
            instance.prediction_stats = data['prediction_stats']
            instance.target_stats = data['target_stats']
            instance.error_stats = data['error_stats']

            return instance


class RacePredictor:
    """
    Race predictor for making predictions using a trained model.
    """

    def __init__(self, model_path: str, db_name: str = "dev", device: Optional[torch.device] = None):
        """
        Initialize predictor with a trained model.

        Args:
            model_path: Path to the saved model
            db_name: Database name to use
            device: PyTorch device (CPU or GPU)
        """
        self.model_path = Path(model_path)
        self.db_name = db_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model components
        self.static_model = None
        self.transformer_model = None
        self.calibration_model = None
        self.static_scaler = None
        self.seq_scaler = None

        # Model info
        self.model_info = None
        self.transformer_params = None

        # Initialize orchestrator
        config = AppConfig()
        sqlite_path = config.get_sqlite_dbpath(db_name)
        self.orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=sqlite_path
        )

        # Load the model
        self._load_model()

        print(f"Predictor initialized with model from {model_path}")
        print(f"Using database: {db_name}")
        print(f"Device: {self.device}")

        # Print calibration info if available
        if self.calibration_model is not None:
            if hasattr(self.calibration_model, 'error_stats') and self.calibration_model.error_stats:
                print(f"Calibration active: {self.calibration_model.calibration_method}")

    def _load_model(self):
        """Load the saved model components."""
        # Check if model path exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path {self.model_path} does not exist")

        # Load static model if available
        static_model_path = self.model_path / "static_model.txt"
        if static_model_path.exists():
            import lightgbm as lgb
            self.static_model = lgb.Booster(model_file=str(static_model_path))
            print("Loaded static model")

        # Load static scaler
        static_scaler_path = self.model_path / "static_scaler.pt"
        if static_scaler_path.exists():
            self.static_scaler = torch.load(static_scaler_path)
            print("Loaded static scaler")

        # Load model info if available
        model_info_path = self.model_path / "model_info.json"
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                self.model_info = json.load(f)

        # Load transformer model if available
        transformer_model_path = self.model_path / "transformer_model.pt"
        transformer_params_path = self.model_path / "transformer_params.json"
        seq_scaler_path = self.model_path / "seq_scaler.pt"

        if transformer_model_path.exists() and transformer_params_path.exists():
            # Load transformer parameters
            with open(transformer_params_path, 'r') as f:
                self.transformer_params = json.load(f)

            # Transformer model will be initialized when needed
            # We need to know the input dimensions first
            print("Loaded transformer parameters")

            # Load sequence scaler if available
            if seq_scaler_path.exists():
                self.seq_scaler = torch.load(seq_scaler_path)
                print("Loaded sequence scaler")

        # Load calibration model if available
        calibration_model_path = self.model_path / "calibration_model.pkl"
        if calibration_model_path.exists():
            self.calibration_model = CalibrationModel.load(calibration_model_path)
            print("Loaded calibration model")

    def _initialize_transformer(self, seq_feature_dim: int, static_feature_dim: int):
        """
        Initialize the transformer model with the correct dimensions.

        Args:
            seq_feature_dim: Sequence feature dimension
            static_feature_dim: Static feature dimension
        """
        if self.transformer_params is None:
            return False

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

    def load_race_data(self, race_id: str = None, race_file: str = None, race_data: Dict = None) -> pd.DataFrame:
        """
        Load race data from various sources.

        Args:
            race_id: Race ID to load from database
            race_file: File path to load race data
            race_data: Direct race data dictionary or list

        Returns:
            DataFrame with race data
        """
        if race_id is not None:
            # Load from database
            df = self.orchestrator.load_historical_races(
                race_filter=None,
                date_filter=None
            )

            # Filter for specified race
            race_data = df[df['comp'] == race_id]

            if len(race_data) == 0:
                raise ValueError(f"Race {race_id} not found in database")

            print(f"Loaded race {race_id} with {len(race_data)} participants from database")

        elif race_file is not None:
            # Load from file
            if race_file.endswith('.json'):
                with open(race_file, 'r') as f:
                    race_data = json.load(f)
            elif race_file.endswith('.csv'):
                race_data = pd.read_csv(race_file)
            else:
                raise ValueError(f"Unsupported file format: {race_file}")

            print(f"Loaded race data from {race_file}")

        elif race_data is not None:
            # Use provided race data
            if isinstance(race_data, dict):
                if 'participants' in race_data:
                    # Handle format with race_info and participants
                    race_df = pd.DataFrame(race_data['participants'])

                    # Add race info to each participant
                    for col, val in race_data.get('race_info', {}).items():
                        if col != 'participants':
                            race_df[col] = val

                    race_data = race_df
                else:
                    # Single participant dict
                    race_data = pd.DataFrame([race_data])
            elif isinstance(race_data, list):
                # List of participants
                race_data = pd.DataFrame(race_data)
        else:
            raise ValueError("No race data source specified")

        # Ensure we have a DataFrame
        if not isinstance(race_data, pd.DataFrame):
            race_data = pd.DataFrame(race_data)

        return race_data

    def prepare_race_data(self, race_data: pd.DataFrame) -> Tuple[
        pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare race data for prediction.

        Args:
            race_data: Race data DataFrame

        Returns:
            Tuple of (static features, sequence features, static sequence features)
            Sequence features will be None if not available
        """
        # Process with orchestrator
        processed_df = self.orchestrator.prepare_features(race_data)

        # Apply embeddings
        embedded_df = self.orchestrator.apply_embeddings(processed_df)

        # Extract static features
        X, _ = self.orchestrator.prepare_training_dataset(embedded_df)

        # Get sequence data if transformer model is available
        X_seq = None
        X_static = None

        if self.transformer_params is not None and hasattr(self.orchestrator, 'prepare_sequence_data'):
            try:
                # Get sequence length from model info or use default
                sequence_length = self.model_info.get('sequence_length', 5) if self.model_info else 5

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
            X: Static features
            X_seq: Sequence features
            X_static: Static features for sequence model
            blend_weight: Weight for static model in blend (0-1)

        Returns:
            NumPy array with predictions
        """
        # Scale static features
        if self.static_scaler is not None:
            X_scaled = self.static_scaler.transform(X)
        else:
            X_scaled = X.values

        # Get static model predictions
        if self.static_model is not None:
            static_preds = self.static_model.predict(X_scaled)
        else:
            static_preds = None

        # Get transformer model predictions if available
        if self.transformer_model is not None and X_seq is not None and X_static is not None:
            # Scale sequence data
            if self.seq_scaler is not None:
                batch_size, seq_len, feat_dim = X_seq.shape
                X_seq_reshaped = X_seq.reshape(batch_size * seq_len, feat_dim)
                X_seq_scaled = self.seq_scaler.transform(X_seq_reshaped)
                X_seq_scaled = X_seq_scaled.reshape(batch_size, seq_len, feat_dim)
            else:
                X_seq_scaled = X_seq

            # Scale static data for transformer
            if self.static_scaler is not None:
                X_static_scaled = self.static_scaler.transform(X_static)
            else:
                X_static_scaled = X_static

            # Get predictions
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
            # Use static predictions
            blended_preds = static_preds if static_preds is not None else np.zeros(len(X))

        # Apply calibration if available
        if self.calibration_model is not None:
            calibrated_preds = self.calibration_model.calibrate(blended_preds)
            return calibrated_preds
        else:
            return blended_preds

    def predict_race(self, race_data: Union[str, Dict, pd.DataFrame], blend_weight: float = 0.7) -> pd.DataFrame:
        """
        Predict race outcome.

        Args:
            race_data: Race ID, file path, or data dictionary/DataFrame
            blend_weight: Weight for static model in blend (0-1)

        Returns:
            DataFrame with predictions
        """
        # Load race data if not a DataFrame
        if isinstance(race_data, str):
            if os.path.exists(race_data):
                race_df = self.load_race_data(race_file=race_data)
            else:
                race_df = self.load_race_data(race_id=race_data)
        elif isinstance(race_data, (dict, list)):
            race_df = self.load_race_data(race_data=race_data)
        else:
            race_df = race_data

        # Save original DataFrame to add predictions to
        original_df = race_df.copy()

        # Prepare data
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

    def predict_races(self, races_data: List[Dict]) -> List[pd.DataFrame]:
        """
        Predict outcomes for multiple races.

        Args:
            races_data: List of race data dicts, each with 'race_info' and 'participants'

        Returns:
            List of DataFrames with predictions for each race
        """
        results = []

        for race_data in races_data:
            race_results = self.predict_race(race_data)
            results.append(race_results)

        return results

    def format_prediction_output(self, prediction_df: pd.DataFrame, output_format: str = 'full',
                                 show_top_n: int = None) -> Union[pd.DataFrame, Dict, str]:
        """
        Format prediction results in various formats.

        Args:
            prediction_df: DataFrame with predictions
            output_format: 'full', 'simple', 'json', or 'markdown'
            show_top_n: Number of top predictions to include

        Returns:
            Formatted prediction output
        """
        # Limit to top N if specified
        if show_top_n is not None:
            prediction_df = prediction_df.head(show_top_n)

        if output_format == 'simple':
            # Simple format with just key columns
            simple_cols = ['cheval', 'predicted_rank', 'predicted_position']
            # Add number/numero if available
            if 'numero' in prediction_df.columns:
                simple_cols = ['numero'] + simple_cols

            return prediction_df[simple_cols]

        elif output_format == 'json':
            # Convert to JSON format
            result = {
                'race_id': prediction_df['comp'].iloc[0] if 'comp' in prediction_df.columns else None,
                'predictions': prediction_df.to_dict(orient='records'),
                'calibrated': self.calibration_model is not None
            }
            return result

        elif output_format == 'markdown':
            # Generate markdown table
            md_cols = ['predicted_rank', 'numero', 'cheval', 'predicted_position']
            # Filter to only include columns that exist
            md_cols = [col for col in md_cols if col in prediction_df.columns]

            # Create header
            header = ' | '.join(md_cols)
            separator = ' | '.join(['---'] * len(md_cols))

            # Create rows
            rows = []
            for _, row in prediction_df.iterrows():
                row_values = [str(row[col]) for col in md_cols]
                rows.append(' | '.join(row_values))

            # Add calibration note if applied
            calibration_note = "\n\n*Predictions are calibrated*" if self.calibration_model is not None else ""

            # Combine into markdown table
            markdown = f"| {header} |\n| {separator} |\n" + '\n'.join([f"| {row} |" for row in rows]) + calibration_note
            return markdown

        else:
            # Full format (default)
            return prediction_df


def main():
    """Main function to parse arguments and run prediction."""
    parser = argparse.ArgumentParser(description="Make race predictions using a trained model")

    # Model and database parameters
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--db', type=str, default="dev", help='Database to use')

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--race-id', type=str, help='Race ID to predict')
    input_group.add_argument('--race-file', type=str, help='JSON or CSV file with race data')

    # Prediction parameters
    parser.add_argument('--blend-weight', type=float, default=0.7,
                        help='Weight for static model in ensemble blend (0-1)')
    parser.add_argument('--no-calibration', action='store_true',
                        help='Disable calibration even if available')

    # Output options
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--format', type=str, choices=['full', 'simple', 'json', 'markdown'],
                        default='full', help='Output format')
    parser.add_argument('--top-n', type=int, help='Show only top N predictions')

    args = parser.parse_args()

    # Initialize predictor
    predictor = RacePredictor(args.model_path, args.db)

    # Disable calibration if requested
    if args.no_calibration and predictor.calibration_model is not None:
        print("Calibration disabled by user")
        predictor.calibration_model = None

    # Get prediction input
    if args.race_id:
        prediction_input = args.race_id
    else:
        prediction_input = args.race_file

    # Make predictions
    results = predictor.predict_race(prediction_input, args.blend_weight)

    # Format output
    formatted_results = predictor.format_prediction_output(results, args.format, args.top_n)

    # Display results
    if args.format == 'json':
        print(json.dumps(formatted_results, indent=2))
    elif args.format == 'markdown':
        print(formatted_results)
    else:
        print("\nRace prediction results:")
        print(formatted_results)

    # Save to file if requested
    if args.output:
        if args.format == 'json':
            with open(args.output, 'w') as f:
                json.dump(formatted_results, f, indent=2)
        elif args.format == 'markdown':
            with open(args.output, 'w') as f:
                f.write(formatted_results)
        else:
            formatted_results.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()