import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging
import joblib
from pathlib import Path
import os
import json


class ErrorCorrectionModel:
    """
    Model that learns to predict the error in base model predictions.
    This model is designed to work with the existing feature pipeline.
    """
    
    def __init__(self, model_type='random_forest', params=None):
        """
        Initialize an error correction model.
        
        Args:
            model_type: Type of model to use ('random_forest', 'linear', etc.)
            params: Parameters for the model
        """
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.feature_columns = None
        self.logger = logging.getLogger("ErrorCorrectionModel")
    
    def fit(self, X: pd.DataFrame, actual_positions: np.ndarray, 
            predicted_positions: np.ndarray) -> Dict[str, Any]:
        """
        Train the error correction model using preprocessed features.
        
        Args:
            X: Preprocessed features (should already be embedded and cleaned)
            actual_positions: Actual race positions
            predicted_positions: Predicted race positions
            
        Returns:
            Dictionary with training results
        """
        # Calculate errors (what we want to predict)
        errors = actual_positions - predicted_positions
        
        # Store info about the errors for diagnostics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        
        self.logger.info(f"Error statistics - Mean: {mean_error:.4f}, Median: {median_error:.4f}, StdDev: {std_error:.4f}")
        
        # Add predicted position as a feature
        X_with_pred = X.copy()
        X_with_pred['predicted_position'] = predicted_positions
        
        # Store feature columns for future use
        self.feature_columns = list(X_with_pred.columns)
        
        # Split data for training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_pred, errors, test_size=0.25, random_state=42
        )
        
        # Create model based on specified type
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=self.params.get('n_estimators', 50),
                max_depth=self.params.get('max_depth', 10),
                min_samples_leaf=self.params.get('min_samples_leaf', 5),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate performance
        test_errors = y_test  # Original errors on test set
        predicted_errors = self.model.predict(X_test)  # Predicted errors
        
        # Calculate corrected_errors = actual_errors - predicted_errors
        # If our error prediction is perfect, corrected_errors should be close to zero
        corrected_errors = test_errors - predicted_errors
        
        # Calculate metrics
        original_mse = np.mean(test_errors**2)
        original_mae = np.mean(np.abs(test_errors))
        corrected_mse = np.mean(corrected_errors**2)
        corrected_mae = np.mean(np.abs(corrected_errors))
        
        # Calculate improvement percentages
        mse_improvement = ((original_mse - corrected_mse) / original_mse) * 100
        mae_improvement = ((original_mae - corrected_mae) / original_mae) * 100
        avg_improvement = (mse_improvement + mae_improvement) / 2
        
        # Log results
        self.logger.info(f"Original error - MSE: {original_mse:.4f}, MAE: {original_mae:.4f}")
        self.logger.info(f"Corrected error - MSE: {corrected_mse:.4f}, MAE: {corrected_mae:.4f}")
        self.logger.info(f"Improvement - MSE: {mse_improvement:.2f}%, MAE: {mae_improvement:.2f}%")
        
        # Get feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_
            
            # Create a sorted list of (feature, importance) tuples
            importance_pairs = sorted(
                zip(self.feature_columns, importance_values),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Store as dictionary
            feature_importance = {name: float(importance) for name, importance in importance_pairs}
            
            # Log top features
            self.logger.info("Top 5 features for error prediction:")
            for feature, importance in importance_pairs[:5]:
                self.logger.info(f"  {feature}: {importance:.4f}")
        
        # Return performance metrics
        return {
            "error_stats": {
                "mean_error": float(mean_error),
                "median_error": float(median_error),
                "std_error": float(std_error)
            },
            "performance": {
                "original_mse": float(original_mse),
                "corrected_mse": float(corrected_mse),
                "original_mae": float(original_mae),
                "corrected_mae": float(corrected_mae),
                "mse_improvement": float(mse_improvement),
                "mae_improvement": float(mae_improvement),
                "avg_improvement": float(avg_improvement)
            },
            "feature_importance": feature_importance
        }
    
    def predict(self, X: pd.DataFrame, predicted_positions: np.ndarray) -> np.ndarray:
        """
        Predict errors for the given features and predictions.
        
        Args:
            X: Preprocessed features (should already be embedded and cleaned)
            predicted_positions: Base model predictions
            
        Returns:
            Predicted errors
        """
        if self.model is None:
            raise ValueError("Model has not been trained")
        
        # Add predicted position as a feature
        X_with_pred = X.copy()
        X_with_pred['predicted_position'] = predicted_positions
        
        # Ensure we have all required feature columns
        missing_cols = set(self.feature_columns) - set(X_with_pred.columns)
        extra_cols = set(X_with_pred.columns) - set(self.feature_columns)
        
        if missing_cols:
            # Add missing columns with zeros
            for col in missing_cols:
                X_with_pred[col] = 0
                
        # Select only the features the model knows about
        X_for_prediction = X_with_pred[self.feature_columns]
        
        # Predict errors
        return self.model.predict(X_for_prediction)
    
    def save(self, filepath: str) -> None:
        """
        Save the error correction model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'params': self.params
        }
        
        joblib.dump(save_dict, filepath)
        self.logger.info(f"Saved error correction model to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ErrorCorrectionModel':
        """
        Load an error correction model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ErrorCorrectionModel
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        save_dict = joblib.load(filepath)
        
        model = cls(
            model_type=save_dict['model_type'],
            params=save_dict['params']
        )
        
        model.model = save_dict['model']
        model.feature_columns = save_dict['feature_columns']
        
        return model


class EnhancedRegressionPredictor:
    """
    Enhanced predictor that combines base models with error correction.
    Integrates with the existing prediction pipeline.
    """
    
    def __init__(self, base_model, error_models: Dict[str, ErrorCorrectionModel]):
        """
        Initialize the enhanced predictor.
        
        Args:
            base_model: Base prediction model (RF or LSTM)
            error_models: Dictionary mapping race types to error correction models
        """
        self.base_model = base_model
        self.error_models = error_models
        self.logger = logging.getLogger("EnhancedRegressionPredictor")
    
    def predict(self, X: pd.DataFrame, race_type: str = None) -> np.ndarray:
        """
        Generate enhanced predictions with error correction.
        
        Args:
            X: Preprocessed features (should already be embedded and cleaned)
            race_type: Optional race type for specialized models
            
        Returns:
            Error-corrected predictions
        """
        # Get base predictions
        base_predictions = self.base_model.predict(X)
        
        # Select appropriate error model
        if race_type and race_type in self.error_models:
            error_model = self.error_models[race_type]
            self.logger.info(f"Using race-specific error model for {race_type}")
        else:
            error_model = self.error_models.get('global')
            self.logger.info("Using global error model")
        
        # If no error model available, return base predictions
        if error_model is None:
            return base_predictions
        
        # Predict expected errors
        predicted_errors = error_model.predict(X, base_predictions)
        
        # Correct predictions by adding the predicted error
        # Note: if error > 0, our prediction was too low; if error < 0, our prediction was too high
        corrected_predictions = base_predictions + predicted_errors
        
        # Ensure all positions are at least 1.0
        corrected_predictions = np.maximum(corrected_predictions, 1.0)
        
        return corrected_predictions
    
    def save(self, base_path: str) -> Dict[str, str]:
        """
        Save the enhanced predictor.
        
        Args:
            base_path: Base directory to save models
            
        Returns:
            Dictionary with paths to saved components
        """
        base_path = Path(base_path)
        os.makedirs(base_path, exist_ok=True)
        
        # Save base model - assumes it's a sklearn model
        base_model_path = base_path / "base_model.joblib"
        joblib.dump(self.base_model, base_model_path)
        
        # Save error models
        error_models_dir = base_path / "error_models"
        os.makedirs(error_models_dir, exist_ok=True)
        
        error_model_paths = {}
        for race_type, model in self.error_models.items():
            model_path = error_models_dir / f"error_model_{race_type}.joblib"
            model.save(str(model_path))
            error_model_paths[race_type] = str(model_path)
        
        # Save model index for easier loading
        model_index = {
            'base_model': str(base_model_path),
            'error_models': error_model_paths,
            'race_types': list(self.error_models.keys())
        }
        
        index_path = base_path / "model_index.json"
        with open(index_path, 'w') as f:
            json.dump(model_index, f, indent=2)
        
        return {
            'base_path': str(base_path),
            'base_model': str(base_model_path),
            'error_models_dir': str(error_models_dir),
            'model_index': str(index_path)
        }
    
    @classmethod
    def load(cls, base_path: str) -> 'EnhancedRegressionPredictor':
        """
        Load an enhanced predictor from disk.
        
        Args:
            base_path: Path where the models are saved
            
        Returns:
            Loaded EnhancedRegressionPredictor
        """
        base_path = Path(base_path)
        
        # Load model index
        index_path = base_path / "model_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Model index not found: {index_path}")
        
        with open(index_path, 'r') as f:
            model_index = json.load(f)
        
        # Load base model
        base_model_path = model_index['base_model']
        base_model = joblib.load(base_model_path)
        
        # Load error models
        error_models = {}
        for race_type in model_index['race_types']:
            model_path = model_index['error_models'][race_type]
            error_models[race_type] = ErrorCorrectionModel.load(model_path)
        
        return cls(base_model, error_models)


def build_error_models_from_preprocessed(preprocessed_df: pd.DataFrame,
                                         race_type_col: str = 'typec',
                                         min_samples_per_type: int = 100) -> Dict[str, Any]:
    """
    Build error correction models from already preprocessed data.
    This function assumes the data has already gone through embedding and feature transformation.
    
    Args:
        preprocessed_df: DataFrame with processed features, actual_position and predicted_position
        race_type_col: Name of column containing race type
        min_samples_per_type: Minimum samples required for race-specific models
        
    Returns:
        Dictionary with error models and performance metrics
    """
    logger = logging.getLogger("build_error_models")
    
    # Check for required columns
    required_cols = ['predicted_position', 'actual_position']
    for col in required_cols:
        if col not in preprocessed_df.columns:
            logger.error(f"Required column missing: {col}")
            return {
                "status": "error",
                "message": f"Required column missing: {col}"
            }
    
    # Make a copy of the DataFrame
    df = preprocessed_df.copy()
    
    # Remove non-numeric columns (except race type if needed)
    cols_to_keep = [race_type_col] if race_type_col in df.columns else []
    cols_to_keep.extend(['predicted_position', 'actual_position'])
    
    # Find numeric columns to keep
    for col in df.columns:
        if col not in cols_to_keep and pd.api.types.is_numeric_dtype(df[col]):
            cols_to_keep.append(col)
    
    # Keep only selected columns
    df = df[cols_to_keep]
    
    # Replace any remaining NaN values with 0
    df = df.fillna(0)
    
    # Build global model
    logger.info(f"Building global error model with {len(df)} samples")
    
    global_model = ErrorCorrectionModel()
    global_results = global_model.fit(
        df.drop(columns=['actual_position', 'predicted_position'] + 
                ([race_type_col] if race_type_col in df.columns else [])),
        df['actual_position'].values,
        df['predicted_position'].values
    )
    
    # Build race-type specific models if possible
    type_models = {}
    type_results = {}
    
    if race_type_col in df.columns:
        for race_type, group in df.groupby(race_type_col):
            if len(group) >= min_samples_per_type:
                logger.info(f"Building model for race type {race_type} with {len(group)} samples")
                
                type_model = ErrorCorrectionModel()
                type_result = type_model.fit(
                    group.drop(columns=['actual_position', 'predicted_position', race_type_col]),
                    group['actual_position'].values,
                    group['predicted_position'].values
                )
                
                type_models[race_type] = type_model
                type_results[race_type] = type_result
    
    # Build full error model dictionary
    error_models = {'global': global_model}
    error_models.update(type_models)
    
    # Collect all results
    return {
        "status": "success",
        "models": error_models,
        "global_results": global_results,
        "type_results": type_results,
        "sample_counts": {
            "global": len(df),
            "by_type": {race_type: len(group) for race_type, group in df.groupby(race_type_col)} 
                      if race_type_col in df.columns else {}
        }
    }


## Updated method for RegressionEnhancer class
def build_correction_models(self, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build error correction models to reduce systematic prediction errors.
    This method integrates with the existing data pipeline.
    
    Args:
        df: DataFrame with prediction data including actual_position and predicted_position
        
    Returns:
        Dictionary with correction models and performance improvements
    """
    from model_training.regressions.error_correction import build_error_models_from_preprocessed
    
    self.logger.info("\n===== BUILDING ERROR CORRECTION MODELS =====")
    
    # Check for required columns
    required_cols = ['predicted_position', 'actual_position']
    for col in required_cols:
        if col not in df.columns:
            self.logger.error(f"Required column missing: {col}")
            return {
                "status": "error",
                "message": f"Required column missing: {col}"
            }
    
    try:
        # 1. Use existing orchestrator to preprocess the data if needed
        if not any(col.startswith(('horse_emb_', 'jockey_emb_', 'couple_emb_')) for col in df.columns):
            self.logger.info("Embeddings not found in data. Applying feature preprocessing...")
            
            # Prepare data with feature engineering and embeddings
            processed_df = self.orchestrator.prepare_features(df)
            processed_df = self.orchestrator.apply_embeddings(
                processed_df, 
                clean_after_embedding=True,
                keep_identifiers=True
            )
            
            # Ensure we have the predicted and actual positions
            processed_df['predicted_position'] = df['predicted_position']
            processed_df['actual_position'] = df['actual_position']
            
            # Keep race type for specialized models
            if 'typec' in df.columns:
                processed_df['typec'] = df['typec']
        else:
            self.logger.info("Embeddings found in data. Using as-is...")
            processed_df = df
        
        # 2. Remove any problematic columns (non-numeric, dates, etc.)
        numeric_cols = ['predicted_position', 'actual_position']
        if 'typec' in processed_df.columns:
            numeric_cols.append('typec')
            
        # Add all numeric columns
        for col in processed_df.columns:
            if col not in numeric_cols and pd.api.types.is_numeric_dtype(processed_df[col]):
                numeric_cols.append(col)
        
        # Keep only numeric columns
        clean_df = processed_df[numeric_cols].copy()
        
        # Handle any remaining NaN values
        clean_df = clean_df.fillna(0)
        
        self.logger.info(f"Prepared data with {len(clean_df)} rows and {len(clean_df.columns)} features")
        
        # 3. Build error correction models
        result = build_error_models_from_preprocessed(
            clean_df,
            race_type_col='typec' if 'typec' in clean_df.columns else None,
            min_samples_per_type=100
        )
        
        if result['status'] == 'success':
            # Store models for later use
            self.error_models = result['models']
            
            # Calculate overall improvement
            global_improvement = result['global_results']['performance']['avg_improvement']
            
            type_improvements = {
                race_type: results['performance']['avg_improvement'] 
                for race_type, results in result['type_results'].items()
            }
            
            # Log improvements
            self.logger.info(f"Global error correction model improvement: {global_improvement:.2f}%")
            
            if type_improvements:
                avg_type_improvement = sum(type_improvements.values()) / len(type_improvements)
                self.logger.info(f"Average race-type specific improvement: {avg_type_improvement:.2f}%")
            
            return {
                "status": "success",
                "models_built": 1 + len(type_improvements),
                "global_model": result['global_results'],
                "type_models": result['type_results'],
                "improvement_summary": {
                    "global": global_improvement,
                    "by_type": type_improvements
                },
                "error_models": self.error_models
            }
        else:
            return result
    
    except Exception as e:
        self.logger.error(f"Error building correction models: {str(e)}")
        import traceback
        self.logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "message": f"Error building correction models: {str(e)}"
        }


## Updated method for creating the enhanced model
def create_enhanced_model(self, blend_weight: float = 0.5) -> Dict[str, Any]:
    """
    Create an enhanced model that combines the base model with error correction.
    This integrates with the existing prediction pipeline.
    
    Args:
        blend_weight: Not used in this implementation (kept for API compatibility)
        
    Returns:
        Dictionary with enhanced model information
    """
    from model_training.regressions.error_correction import EnhancedRegressionPredictor
    
    if not hasattr(self, 'error_models') or not self.error_models:
        return {
            "status": "error", 
            "message": "No error correction models available"
        }
    
    if self.rf_model is None:
        return {
            "status": "error",
            "message": "No base model available"
        }
    
    self.logger.info("Creating enhanced prediction model with error correction")
    
    # Create enhanced predictor
    enhanced_model = EnhancedRegressionPredictor(
        base_model=self.rf_model,
        error_models=self.error_models
    )
    
    # Save the model
    enhanced_dir = self.output_dir / 'enhanced_model'
    
    saved_paths = enhanced_model.save(str(enhanced_dir))
    
    # Save metadata
    metadata = {
        "model_type": "EnhancedRegressionPredictor",
        "base_model_version": self.model_config.get('version', 'unknown'),
        "error_models": list(self.error_models.keys()),
        "created_at": datetime.now().isoformat(),
        "paths": saved_paths
    }
    
    with open(self.output_dir / 'enhanced_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    self.logger.info(f"Created and saved enhanced model with {len(self.error_models)} error correction models")
    
    return {
        "status": "success",
        "model_path": saved_paths['base_path'],
        "error_models_dir": saved_paths['error_models_dir'],
        "metadata": metadata
    }