"""
Configurable prediction blending module for horse race predictions.

This module implements race-specific blending of RF, LSTM, and TabNet model predictions
based on race characteristics defined in config.yaml.
"""

import yaml
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path


class PredictionBlender:
    """
    Handles configurable blending of model predictions based on race characteristics.
    
    Features:
    - Fail-fast validation of configuration and predictions
    - Race-specific weight selection based on conditions
    - Transparent logging of applied weights
    - Scalable rule-based system
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the blender with configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        self.config_path = Path(config_path)
        
        # Required model names (set before validation)
        self.required_models = {'rf', 'lstm', 'tabnet'}
        
        # Fail-fast: Check config file exists
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        # Load and validate configuration
        self._load_config()
        self._validate_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Extract model_blending section
            if 'model_blending' not in config:
                raise ValueError("Configuration must contain 'model_blending' section")
                
            self.blending_config = config['model_blending']
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    def _validate_config(self):
        """
        Validate configuration structure and weights.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        if 'default' not in self.blending_config:
            raise ValueError("Configuration must contain 'default' weights")
            
        # Validate default weights
        default_weights = self.blending_config['default']
        self._validate_weights(default_weights, "default")
        
        # Validate rule weights if rules exist
        if 'rules' in self.blending_config:
            for rule_name, rule in self.blending_config['rules'].items():
                if 'weights' not in rule:
                    raise ValueError(f"Rule '{rule_name}' must contain 'weights' section")
                    
                if 'condition' not in rule:
                    raise ValueError(f"Rule '{rule_name}' must contain 'condition' section")
                    
                self._validate_weights(rule['weights'], f"rule '{rule_name}'")

    def _validate_weights(self, weights: Dict[str, float], context: str):
        """
        Validate that weights are correct for a given context.
        
        Args:
            weights: Dictionary of model weights
            context: Description for error messages
            
        Raises:
            ValueError: If weights are invalid
        """
        # Check all required models present
        missing_models = self.required_models - set(weights.keys())
        if missing_models:
            raise ValueError(f"Missing model weights in {context}: {missing_models}")
            
        # Check for unexpected models
        extra_models = set(weights.keys()) - self.required_models
        if extra_models:
            raise ValueError(f"Unexpected model weights in {context}: {extra_models}")
            
        # Check all weights are positive
        for model, weight in weights.items():
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Invalid weight for {model} in {context}: {weight} (must be positive number)")
                
        # Check weights sum to 1.0 (with tolerance for floating point)
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weights in {context} must sum to 1.0, got {weight_sum:.4f}: {weights}")

    def get_blend_weights(self, race_metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Get blend weights based on race metadata.
        
        Args:
            race_metadata: Dictionary containing race characteristics:
                - typec: Race type (e.g., 'P', 'T')
                - distance: Race distance in meters
                - partant: Number of horses in race
                - etc.
                
        Returns:
            Dictionary with model weights: {'rf': float, 'lstm': float, 'tabnet': float}
            
        Raises:
            ValueError: If race_metadata is invalid
        """
        # Fail-fast: Validate race metadata
        if not isinstance(race_metadata, dict):
            raise ValueError(f"race_metadata must be dict, got {type(race_metadata)}")
            
        # Check for rules if they exist
        if 'rules' in self.blending_config:
            for rule_name, rule in self.blending_config['rules'].items():
                if self._matches_condition(race_metadata, rule['condition']):
                    weights = rule['weights'].copy()
                    # Log which rule was applied
                    weights['_applied_rule'] = rule_name
                    return weights
        
        # Return default weights if no rule matches
        weights = self.blending_config['default'].copy()
        weights['_applied_rule'] = 'default'
        return weights

    def _matches_condition(self, race_metadata: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """
        Check if race metadata matches a condition.
        
        Args:
            race_metadata: Race characteristics
            condition: Condition to check
            
        Returns:
            True if all conditions are met
        """
        for key, value in condition.items():
            if key.endswith('_min'):
                # Minimum value condition
                field = key[:-4]  # Remove '_min' suffix
                if field not in race_metadata:
                    return False
                if race_metadata[field] < value:
                    return False
                    
            elif key.endswith('_max'):
                # Maximum value condition
                field = key[:-4]  # Remove '_max' suffix
                if field not in race_metadata:
                    return False
                if race_metadata[field] > value:
                    return False
                    
            else:
                # Exact match condition
                if key not in race_metadata:
                    return False
                    
                # Handle list of acceptable values
                if isinstance(value, list):
                    if race_metadata[key] not in value:
                        return False
                else:
                    if race_metadata[key] != value:
                        return False
        
        return True

    def blend_predictions(self, predictions_dict: Dict[str, np.ndarray], 
                         race_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Blend model predictions using race-specific weights.
        
        Args:
            predictions_dict: Dictionary with model predictions:
                {'rf': np.array, 'lstm': np.array, 'tabnet': np.array}
            race_metadata: Race characteristics for weight selection
            
        Returns:
            Dictionary containing:
                - 'blended_predictions': Final blended predictions
                - 'weights_used': Weights applied
                - 'applied_rule': Rule that was applied
                - 'individual_predictions': Original predictions
                
        Raises:
            ValueError: If predictions are invalid or misaligned
        """
        # Fail-fast validation
        self._validate_predictions(predictions_dict)
        
        # Get appropriate weights
        weights = self.get_blend_weights(race_metadata)
        applied_rule = weights.pop('_applied_rule')  # Remove metadata
        
        # Convert predictions to numpy arrays
        rf_preds = np.array(predictions_dict['rf'])
        lstm_preds = np.array(predictions_dict['lstm'])
        tabnet_preds = np.array(predictions_dict['tabnet'])
        
        # Blend predictions
        blended_preds = (
            rf_preds * weights['rf'] + 
            lstm_preds * weights['lstm'] + 
            tabnet_preds * weights['tabnet']
        )
        
        return {
            'blended_predictions': blended_preds,
            'weights_used': weights,
            'applied_rule': applied_rule,
            'individual_predictions': {
                'rf': rf_preds,
                'lstm': lstm_preds,  
                'tabnet': tabnet_preds
            }
        }

    def _validate_predictions(self, predictions_dict: Dict[str, np.ndarray]):
        """
        Validate prediction dictionary structure and alignment.
        
        Args:
            predictions_dict: Dictionary with model predictions
            
        Raises:
            ValueError: If predictions are invalid
        """
        # Check all required models present
        missing_models = self.required_models - set(predictions_dict.keys())
        if missing_models:
            raise ValueError(f"Missing model predictions: {missing_models}")
            
        # Check for unexpected models
        extra_models = set(predictions_dict.keys()) - self.required_models
        if extra_models:
            raise ValueError(f"Unexpected model predictions: {extra_models}")
            
        # Convert to arrays and check lengths
        lengths = []
        for model, preds in predictions_dict.items():
            if preds is None:
                raise ValueError(f"Predictions for {model} cannot be None")
                
            pred_array = np.array(preds)
            if pred_array.size == 0:
                raise ValueError(f"Predictions for {model} cannot be empty")
                
            lengths.append(len(pred_array))
            
        # Check all prediction arrays have same length
        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                f"Prediction lengths misaligned: "
                f"rf={lengths[0]}, lstm={lengths[1]}, tabnet={lengths[2]}"
            )

    def get_available_rules(self) -> List[str]:
        """
        Get list of available blending rules.
        
        Returns:
            List of rule names
        """
        if 'rules' not in self.blending_config:
            return []
        return list(self.blending_config['rules'].keys())

    def describe_rule(self, rule_name: str) -> Dict[str, Any]:
        """
        Get detailed description of a specific rule.
        
        Args:
            rule_name: Name of the rule to describe
            
        Returns:
            Dictionary with rule details
            
        Raises:
            ValueError: If rule doesn't exist
        """
        if 'rules' not in self.blending_config or rule_name not in self.blending_config['rules']:
            available = self.get_available_rules()
            raise ValueError(f"Rule '{rule_name}' not found. Available rules: {available}")
            
        return self.blending_config['rules'][rule_name].copy()