import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Any, Optional

from model_training.regressions.error_correction import EnhancedRegressionPredictor


class EnhancedRacePredictor:
    """
    Race predictor that uses error correction to enhance predictions.
    """

    def __init__(self, base_model_path, error_models_dir):
        """
        Initialize the enhanced race predictor.

        Args:
            base_model_path: Path to base model
            error_models_dir: Directory with error models
        """
        self.predictor = EnhancedPredictor.load(base_model_path, error_models_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict race outcomes with error correction.

        Args:
            race_df: DataFrame with race data

        Returns:
            DataFrame with enhanced predictions
        """
        # Extract race type if available
        race_type = None
        if 'typec' in race_df.columns:
            # Use the most common race type in the DataFrame
            race_type = race_df['typec'].mode().iloc[0]

        # Generate enhanced predictions
        enhanced_predictions = self.predictor.predict(race_df, race_type)

        # Add predictions to result DataFrame
        result_df = race_df.copy()
        result_df['predicted_position'] = enhanced_predictions

        # Sort by predicted position
        result_df = result_df.sort_values('predicted_position')

        # Add ranks
        result_df['predicted_rank'] = range(1, len(result_df) + 1)

        # Create arrival string
        numeros_ordered = result_df['numero'].astype(str).tolist()
        predicted_arriv = '-'.join(numeros_ordered)
        result_df['predicted_arriv'] = predicted_arriv

        return result_df