from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error


class BaseModel(ABC):
    """
    Abstract base class for horse racing prediction models.
    
    All alternative models should inherit from this class and implement
    the required methods for training, prediction, and evaluation.
    """
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration dictionary
            verbose: Whether to print verbose output
        """
        self.config = config
        self.verbose = verbose
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.performance_metrics = {}
        
    @abstractmethod
    def prepare_features(self, X_sequences: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """
        Prepare features for this specific model architecture.
        
        Args:
            X_sequences: Sequential features (batch_size, sequence_length, features)
            X_static: Static features (batch_size, static_features)
            
        Returns:
            Prepared feature array for model input
        """
        pass
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """
        Build the model architecture.
        
        Args:
            input_shape: Shape of input features
            
        Returns:
            Compiled model object
        """
        pass
    
    @abstractmethod
    def train(self, X_sequences: np.ndarray, X_static: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_sequences: Sequential training features
            X_static: Static training features  
            y: Target values (horse positions)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results dictionary
        """
        pass
    
    @abstractmethod
    def predict(self, X_sequences: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_sequences: Sequential features for prediction
            X_static: Static features for prediction
            
        Returns:
            Predicted horse positions
        """
        pass
    
    def evaluate(self, X_sequences: np.ndarray, X_static: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_sequences: Sequential test features
            X_static: Static test features
            y: True target values
            
        Returns:
            Dictionary of performance metrics
        """
        # Make predictions
        y_pred = self.predict(X_sequences, X_static)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Top-3 accuracy (horses predicted in top 3 positions)
        top3_accuracy = self._calculate_top3_accuracy(y, y_pred)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'top3_accuracy': float(top3_accuracy)
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def _calculate_top3_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy for top-3 position predictions."""
        # Consider a prediction correct if actual position is within top 3
        # and predicted position is also within top 3
        top3_true = y_true <= 3
        top3_pred = y_pred <= 3
        
        # Accuracy is intersection over union of top 3 predictions
        correct_top3 = np.logical_and(top3_true, top3_pred)
        total_top3 = np.logical_or(top3_true, top3_pred)
        
        if total_top3.sum() == 0:
            return 0.0
        
        return correct_top3.sum() / total_top3.sum()
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        import joblib
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics,
            'model_type': self.__class__.__name__
        }
        
        joblib.dump(model_data, path)
        
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model.
        
        Args:
            path: Path to load the model from
        """
        import joblib
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.training_history = model_data.get('training_history', {})
        self.performance_metrics = model_data.get('performance_metrics', {})
        self.is_trained = True
        
        if self.verbose:
            print(f"Model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and performance metrics.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'config': self.config,
            'performance_metrics': self.performance_metrics,
            'training_history': self.training_history
        }
    
    def assert_data_quality(self, X_sequences: np.ndarray, X_static: np.ndarray, y: np.ndarray = None) -> None:
        """
        Assert data quality and shapes.
        
        Args:
            X_sequences: Sequential features
            X_static: Static features
            y: Target values (optional for prediction)
        """
        # Check for NaN values
        assert not np.isnan(X_sequences).any(), "X_sequences contains NaN values"
        assert not np.isnan(X_static).any(), "X_static contains NaN values"
        if y is not None:
            assert not np.isnan(y).any(), "Target y contains NaN values"
        
        # Check shapes match
        assert X_sequences.shape[0] == X_static.shape[0], \
            f"Batch size mismatch: X_sequences={X_sequences.shape[0]}, X_static={X_static.shape[0]}"
        
        if y is not None:
            assert X_sequences.shape[0] == len(y), \
                f"Batch size mismatch: X_sequences={X_sequences.shape[0]}, y={len(y)}"
        
        # Check data types
        assert X_sequences.dtype in [np.float32, np.float64], \
            f"X_sequences must be float, got {X_sequences.dtype}"
        assert X_static.dtype in [np.float32, np.float64], \
            f"X_static must be float, got {X_static.dtype}"
        
        if self.verbose:
            print(f"Data quality checks passed:")
            print(f"  X_sequences shape: {X_sequences.shape}")
            print(f"  X_static shape: {X_static.shape}")
            if y is not None:
                print(f"  y shape: {y.shape}")