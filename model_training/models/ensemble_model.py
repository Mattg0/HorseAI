import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Any, List, Tuple, Optional
import joblib

from .base_model import BaseModel
from .feedforward_model import FeedforwardModel
from .transformer_model import TransformerModel


class EnsembleModel(BaseModel):
    """
    Ensemble stacking model for horse racing prediction.
    
    This model combines predictions from multiple base models (Feedforward,
    RandomForest, Transformer) using a meta-learner. It implements proper
    cross-validation to prevent data leakage and weights models based on
    recent performance.
    """
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the Ensemble model.
        
        Args:
            config: Model configuration dictionary
            verbose: Whether to print verbose output
        """
        super().__init__(config, verbose)
        
        # Extract ensemble-specific config
        self.meta_learner_type = config.get('meta_learner', 'logistic_regression')
        self.cv_folds = config.get('cv_folds', 5)
        self.include_models = config.get('include_models', ['feedforward', 'random_forest', 'transformer'])
        self.stacking_method = config.get('stacking_method', 'blending')
        self.blend_weights = config.get('blend_weights', {
            'feedforward': 0.4,
            'random_forest': 0.4,
            'transformer': 0.2
        })
        
        # Initialize base models
        self.base_models = {}
        self.meta_learner = None
        
        # Performance tracking
        self.model_performances = {}
        
        if self.verbose:
            print(f"EnsembleModel initialized with:")
            print(f"  Meta learner: {self.meta_learner_type}")
            print(f"  CV folds: {self.cv_folds}")
            print(f"  Include models: {self.include_models}")
            print(f"  Stacking method: {self.stacking_method}")
    
    def _initialize_base_models(self, config: Dict[str, Any]) -> None:
        """Initialize base models with their configurations."""
        
        if 'feedforward' in self.include_models:
            ff_config = config.get('feedforward', {})
            self.base_models['feedforward'] = FeedforwardModel(ff_config, verbose=self.verbose)
        
        if 'random_forest' in self.include_models:
            # RandomForest doesn't use neural network config, create simple config
            rf_config = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1
            }
            self.base_models['random_forest'] = RandomForestRegressor(**rf_config)
        
        if 'transformer' in self.include_models:
            transformer_config = config.get('transformer', {})
            self.base_models['transformer'] = TransformerModel(transformer_config, verbose=self.verbose)
    
    def prepare_features(self, X_sequences: np.ndarray, X_static: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for ensemble processing.
        
        Args:
            X_sequences: Sequential features
            X_static: Static features
            
        Returns:
            Prepared sequential and static features
        """
        # Ensemble uses the same features as base models
        return X_sequences, X_static
    
    def _prepare_rf_features(self, X_sequences: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """Prepare features for RandomForest (flatten sequential features)."""
        batch_size, sequence_length, n_features = X_sequences.shape
        X_seq_flat = X_sequences.reshape(batch_size, -1)
        return np.concatenate([X_seq_flat, X_static], axis=1)
    
    def build_model(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build the ensemble model by initializing base models and meta-learner.
        
        Args:
            input_shape: Not used directly, base models handle their own shapes
        """
        # Initialize meta-learner
        if self.meta_learner_type == 'logistic_regression':
            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        else:
            # Default to RandomForest meta-learner
            self.meta_learner = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        
        if self.verbose:
            print(f"Ensemble meta-learner: {type(self.meta_learner).__name__}")
    
    def train(self, X_sequences: np.ndarray, X_static: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the ensemble model using cross-validation stacking.
        
        Args:
            X_sequences: Sequential training features
            X_static: Static training features
            y: Target values (horse positions)
            validation_split: Not used in ensemble (uses CV instead)
            
        Returns:
            Training results dictionary
        """
        # Assert data quality
        self.assert_data_quality(X_sequences, X_static, y)
        
        if self.verbose:
            print(f"Training ensemble model with {len(y)} samples...")
        
        # Initialize base models if not done
        if not self.base_models:
            self._initialize_base_models(self.config)
        
        # Build meta-learner if not done
        if self.meta_learner is None:
            self.build_model(None)
        
        # Prepare features for different model types
        X_seq_prep, X_static_prep = self.prepare_features(X_sequences, X_static)
        X_rf_features = self._prepare_rf_features(X_sequences, X_static)
        
        # Generate base model predictions using cross-validation
        cv_predictions = {}
        base_model_scores = {}
        
        # Set up cross-validation
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for model_name in self.include_models:
            if self.verbose:
                print(f"Training base model: {model_name}")
            
            if model_name == 'random_forest':
                # RandomForest uses flattened features
                model = self.base_models[model_name]
                cv_preds = cross_val_predict(
                    model, X_rf_features, y, 
                    cv=kfold, method='predict'
                )
                
                # Train on full dataset for final model
                model.fit(X_rf_features, y)
                
            else:
                # Neural network models (feedforward, transformer)
                model = self.base_models[model_name]
                cv_preds = np.zeros_like(y, dtype=float)
                
                # Manual cross-validation for neural network models
                for train_idx, val_idx in kfold.split(X_seq_prep):
                    # Split data
                    X_seq_train = X_seq_prep[train_idx]
                    X_static_train = X_static_prep[train_idx]
                    y_train = y[train_idx]
                    
                    X_seq_val = X_seq_prep[val_idx]
                    X_static_val = X_static_prep[val_idx]
                    
                    # Create fresh model instance for this fold
                    if model_name == 'feedforward':
                        fold_model = FeedforwardModel(self.config.get('feedforward', {}), verbose=False)
                    elif model_name == 'transformer':
                        fold_model = TransformerModel(self.config.get('transformer', {}), verbose=False)
                    
                    # Train on fold
                    fold_model.train(X_seq_train, X_static_train, y_train, validation_split=0.0)
                    
                    # Predict on validation set
                    fold_preds = fold_model.predict(X_seq_val, X_static_val)
                    cv_preds[val_idx] = fold_preds
                
                # Train final model on full dataset
                model.train(X_seq_prep, X_static_prep, y, validation_split=0.1)
            
            # Store cross-validation predictions
            cv_predictions[model_name] = cv_preds
            
            # Calculate model performance
            mae = mean_absolute_error(y, cv_preds)
            rmse = np.sqrt(mean_squared_error(y, cv_preds))
            base_model_scores[model_name] = {'mae': mae, 'rmse': rmse}
            
            if self.verbose:
                print(f"  {model_name} CV MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        # Prepare meta-features (base model predictions)
        meta_features = np.column_stack([cv_predictions[name] for name in self.include_models])
        
        # Train meta-learner
        if self.stacking_method == 'stacking':
            # Use meta-learner
            if self.meta_learner_type == 'logistic_regression':
                # For LogisticRegression, convert to classification bins
                y_binned = self._bin_positions(y)
                self.meta_learner.fit(meta_features, y_binned)
            else:
                self.meta_learner.fit(meta_features, y)
            
        elif self.stacking_method == 'blending':
            # Use fixed blend weights (no training needed)
            if self.verbose:
                print("Using fixed blend weights for ensemble")
        
        self.is_trained = True
        self.model_performances = base_model_scores
        
        # Calculate ensemble predictions for performance metrics
        if self.stacking_method == 'stacking':
            if self.meta_learner_type == 'logistic_regression':
                ensemble_preds = self.meta_learner.predict_proba(meta_features)
                # Convert probabilities back to positions (simplified)
                ensemble_preds = np.argmax(ensemble_preds, axis=1) + 1
            else:
                ensemble_preds = self.meta_learner.predict(meta_features)
        else:
            # Blending
            ensemble_preds = np.zeros_like(y, dtype=float)
            for i, model_name in enumerate(self.include_models):
                weight = self.blend_weights.get(model_name, 1.0 / len(self.include_models))
                ensemble_preds += weight * cv_predictions[model_name]
        
        # Calculate final ensemble performance
        ensemble_mae = mean_absolute_error(y, ensemble_preds)
        ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_preds))
        
        results = {
            'status': 'success',
            'model_type': 'ensemble',
            'stacking_method': self.stacking_method,
            'base_models': list(self.include_models),
            'base_model_performances': base_model_scores,
            'ensemble_performance': {
                'mae': float(ensemble_mae),
                'rmse': float(ensemble_rmse)
            },
            'training_samples': len(y),
            'cv_folds': self.cv_folds
        }
        
        if self.verbose:
            print(f"Ensemble training completed:")
            print(f"  Method: {self.stacking_method}")
            print(f"  Ensemble MAE: {ensemble_mae:.4f}")
            print(f"  Ensemble RMSE: {ensemble_rmse:.4f}")
        
        return results
    
    def _bin_positions(self, positions: np.ndarray, num_bins: int = 5) -> np.ndarray:
        """Convert continuous positions to bins for classification."""
        # Create bins: [1-3], [4-6], [7-9], [10-12], [13+]
        bins = [1, 4, 7, 10, 13, np.inf]
        return np.digitize(positions, bins) - 1  # 0-indexed
    
    def predict(self, X_sequences: np.ndarray, X_static: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained ensemble model.
        
        Args:
            X_sequences: Sequential features for prediction
            X_static: Static features for prediction
            
        Returns:
            Predicted horse positions
        """
        # Assert model is trained
        assert self.is_trained, "Model must be trained before making predictions"
        assert self.base_models, "Base models are not available"
        
        # Assert data quality
        self.assert_data_quality(X_sequences, X_static)
        
        # Prepare features
        X_seq_prep, X_static_prep = self.prepare_features(X_sequences, X_static)
        X_rf_features = self._prepare_rf_features(X_sequences, X_static)
        
        # Get predictions from base models
        base_predictions = {}
        
        for model_name in self.include_models:
            if model_name == 'random_forest':
                preds = self.base_models[model_name].predict(X_rf_features)
            else:
                preds = self.base_models[model_name].predict(X_seq_prep, X_static_prep)
            
            base_predictions[model_name] = preds
        
        # Combine predictions
        if self.stacking_method == 'stacking' and self.meta_learner is not None:
            # Use meta-learner
            meta_features = np.column_stack([base_predictions[name] for name in self.include_models])
            
            if self.meta_learner_type == 'logistic_regression':
                # Get class probabilities and convert to positions
                probs = self.meta_learner.predict_proba(meta_features)
                predictions = np.argmax(probs, axis=1) + 1
            else:
                predictions = self.meta_learner.predict(meta_features)
        
        else:
            # Use blending
            predictions = np.zeros(len(X_sequences), dtype=float)
            for model_name in self.include_models:
                weight = self.blend_weights.get(model_name, 1.0 / len(self.include_models))
                predictions += weight * base_predictions[model_name]
        
        # Ensure positive values
        predictions = np.maximum(predictions, 1.0)
        
        return predictions
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current model weights based on performance.
        
        Returns:
            Dictionary of model weights
        """
        if not self.model_performances:
            return self.blend_weights
        
        # Calculate weights inversely proportional to MAE
        weights = {}
        total_inverse_mae = 0
        
        for model_name, performance in self.model_performances.items():
            inverse_mae = 1.0 / (performance['mae'] + 1e-6)  # Avoid division by zero
            weights[model_name] = inverse_mae
            total_inverse_mae += inverse_mae
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_inverse_mae
        
        return weights