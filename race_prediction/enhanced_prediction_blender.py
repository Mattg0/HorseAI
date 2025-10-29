"""
Enhanced prediction blending module for horse race predictions.

This module implements race-specific blending of RF and TabNet models based on race characteristics
and model performance, enhanced with competitive field analysis.
"""

import numpy as np
import yaml
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from .competitive_field_analyzer import CompetitiveFieldAnalyzer


class EnhancedPredictionBlender:
    """
    Advanced prediction blender that handles both legacy and alternative models
    with dynamic weight assignment based on race characteristics and model performance.
    """
    
    def __init__(self, config_path: str = 'config.yaml', verbose: bool = False, db_path: str = None):
        """
        Initialize the enhanced blender.

        Args:
            config_path: Path to configuration file
            verbose: Enable verbose logging
            db_path: Database path for historical data queries
        """
        self.config_path = Path(config_path)
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Model categories
        self.legacy_models = {'rf', 'lstm', 'tabnet'}
        self.alternative_models = {'tabnet'}
        self.all_models = self.legacy_models | self.alternative_models

        # Initialize competitive field analyzer with database path
        self.competitive_analyzer = CompetitiveFieldAnalyzer(verbose=verbose, db_path=db_path)

        # Load configuration
        self._load_config()

        # Model weights (will be set by configuration or defaults)
        self.default_weights = {}
        self.race_specific_weights = {}

        self._initialize_weights()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            if self.verbose:
                self.logger.warning(f"Config file {self.config_path} not found, using defaults")
            self.config = {}
    
    def _initialize_weights(self):
        """Initialize default and race-specific weights."""
        # Legacy model weights from blend config
        blend_config = self.config.get('blend', {})
        self.default_weights.update({
            'rf': blend_config.get('rf_weight', 0.8),
            'lstm': blend_config.get('lstm_weight', 0.1), 
            'tabnet': blend_config.get('tabnet_weight', 0.1)
        })
        
        # Alternative model weights (initially 0, can be enabled dynamically)
        alt_config = self.config.get('alternative_models', {})
        self.default_weights.update({
            'transformer': alt_config.get('transformer', {}).get('weight', 0.0),
            'ensemble': alt_config.get('ensemble', {}).get('weight', 0.0)
        })
        
        # Race-specific blending rules (if any)
        self.race_specific_weights = self.config.get('blending_rules', {})
    
    def blend_predictions(self, predictions: Dict[str, np.ndarray], 
                         race_metadata: Optional[Dict[str, Any]] = None,
                         model_performance: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Blend predictions from multiple models.
        
        Args:
            predictions: Dictionary of model predictions {model_name: predictions_array}
            race_metadata: Optional race characteristics for adaptive blending
            model_performance: Optional recent performance metrics for models
            
        Returns:
            Tuple of (blended_predictions, blend_info)
        """
        # Filter valid predictions
        valid_predictions = {k: v for k, v in predictions.items() 
                           if v is not None and len(v) > 0}
        
        if not valid_predictions:
            raise ValueError("No valid predictions provided")
        
        # Get weights for this race
        weights = self._get_race_weights(valid_predictions.keys(), race_metadata, model_performance)
        
        # Normalize weights for available models
        total_weight = sum(weights.values())
        if total_weight == 0:
            # Equal weights if no specific weights
            weights = {model: 1.0/len(valid_predictions) for model in valid_predictions}
            total_weight = 1.0
        else:
            weights = {model: w/total_weight for model, w in weights.items()}
        
        # Blend predictions
        prediction_arrays = list(valid_predictions.values())
        weight_arrays = [weights[model] for model in valid_predictions.keys()]
        
        blended = np.zeros_like(prediction_arrays[0])
        for preds, weight in zip(prediction_arrays, weight_arrays):
            blended += preds * weight
        
        # Blend information for debugging/logging
        blend_info = {
            'models_used': list(valid_predictions.keys()),
            'weights_applied': weights,
            'total_models': len(valid_predictions),
            'blend_method': self._get_blend_method(race_metadata)
        }
        
        if self.verbose:
            self.logger.info(f"Blended {len(valid_predictions)} models with weights: {weights}")
        
        return blended, blend_info

    def blend_with_competitive_analysis(self, predictions: Dict[str, np.ndarray],
                                      race_data: Optional[Any] = None,
                                      race_metadata: Optional[Dict[str, Any]] = None,
                                      model_performance: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhanced blending with competitive field analysis.

        Args:
            predictions: Dictionary of model predictions {model_name: predictions_array}
            race_data: Race participant data for competitive analysis
            race_metadata: Optional race characteristics for adaptive blending
            model_performance: Optional recent performance metrics for models

        Returns:
            Tuple of (blended_predictions, enhanced_blend_info)
        """
        # Filter valid predictions for competitive analysis
        valid_predictions = {k: v for k, v in predictions.items()
                           if v is not None and len(v) > 0}

        # Step 1: Basic model blending
        base_blended, base_blend_info = self.blend_predictions(
            predictions, race_metadata, model_performance
        )

        # Step 2: Apply competitive analysis if race data is available
        if race_data is not None and len(race_data) > 0 and valid_predictions:
            if self.verbose:
                self.logger.info("Applying competitive field analysis to enhance predictions")

            # Run competitive analysis with filtered valid predictions
            competitive_results = self.competitive_analyzer.analyze_competitive_field(
                race_data, valid_predictions, race_metadata or {}
            )

            # Extract enhanced predictions
            enhanced_predictions = competitive_results['enhanced_predictions']

            # Blend the enhanced predictions using the same weights as base blending
            valid_enhanced = {k: v for k, v in enhanced_predictions.items()
                            if k in predictions and v is not None}

            if valid_enhanced:
                # Apply the same blending weights to enhanced predictions
                weights = base_blend_info['weights_applied']
                enhanced_blended = np.zeros_like(base_blended)

                for model_name, enhanced_preds in valid_enhanced.items():
                    if model_name in weights:
                        enhanced_blended += enhanced_preds * weights[model_name]

                # Enhanced blend info
                # Note: competitive_analysis is a list, competitive_analysis_summary is a dict
                competitive_summary = competitive_results.get('competitive_analysis_summary', {})

                enhanced_blend_info = {
                    **base_blend_info,
                    'competitive_analysis_applied': True,
                    'competitive_summary': competitive_results.get('adjustment_summary', {}),
                    'competitive_scores': competitive_summary.get('competitive_scores', {}),
                    'field_statistics': competitive_summary.get('field_statistics', {}),
                    'top_contenders': competitive_summary.get('top_contenders', []),
                    'performance_expectations': competitive_results.get('audit_trail', {}).get('performance_expectations', {}),
                    'competitive_analysis_list': competitive_results.get('competitive_analysis', [])  # Per-horse list
                }

                if self.verbose:
                    field_stats = enhanced_blend_info['field_statistics']
                    avg_score = field_stats.get('average_competitive_score', 0)
                    self.logger.info(f"Competitive analysis complete - Field avg competitive score: {avg_score:.3f}")

                return enhanced_blended, enhanced_blend_info

        # Fallback to base blending if no competitive analysis possible
        base_blend_info['competitive_analysis_applied'] = False
        if self.verbose and race_data is None:
            self.logger.info("No race data provided - using base blending only")

        return base_blended, base_blend_info
    
    def _get_race_weights(self, available_models: List[str], 
                         race_metadata: Optional[Dict[str, Any]] = None,
                         model_performance: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Get weights for available models based on race characteristics and performance.
        
        Args:
            available_models: List of available model names
            race_metadata: Race characteristics
            model_performance: Recent performance metrics
            
        Returns:
            Dictionary of weights for each available model
        """
        weights = {}
        
        # Start with default weights
        for model in available_models:
            weights[model] = self.default_weights.get(model, 0.0)
        
        # Apply race-specific adjustments
        if race_metadata:
            weights = self._apply_race_specific_weights(weights, race_metadata)
        
        # Apply performance-based adjustments
        if model_performance:
            weights = self._apply_performance_weights(weights, model_performance)
        
        return weights
    
    def _apply_race_specific_weights(self, weights: Dict[str, float], 
                                   race_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Apply race-specific weight adjustments."""
        # Example rule-based adjustments
        distance = race_metadata.get('distance', 0)
        race_type = race_metadata.get('typec', '')
        field_size = race_metadata.get('field_size', 0)
        
        # Adjust for distance (example logic)
        if distance > 2000:  # Long distance races
            if 'lstm' in weights:
                weights['lstm'] *= 1.2  # LSTM better for sequences
            if 'transformer' in weights:
                weights['transformer'] *= 1.3  # Transformer even better
        elif distance < 1200:  # Sprint races
            if 'rf' in weights:
                weights['rf'] *= 1.2  # RF good for sprints
        
        # Adjust for race type
        if race_type == 'P':  # Plat races
            pass  # No specific adjustments for Plat races
        elif race_type == 'T':  # Trot races
            if 'ensemble' in weights:
                weights['ensemble'] *= 1.2
        
        # Adjust for field size
        if field_size > 16:  # Large fields
            if 'ensemble' in weights:
                weights['ensemble'] *= 1.3  # Ensemble better for complex fields
            if 'transformer' in weights:
                weights['transformer'] *= 1.2
        
        return weights
    
    def _apply_performance_weights(self, weights: Dict[str, float], 
                                 model_performance: Dict[str, float]) -> Dict[str, float]:
        """Apply performance-based weight adjustments."""
        # Boost weights for better performing models
        for model in weights:
            if model in model_performance:
                mae = model_performance[model]
                if mae < 5.0:  # Very good performance
                    weights[model] *= 1.3
                elif mae < 6.0:  # Good performance
                    weights[model] *= 1.1
                elif mae > 8.0:  # Poor performance
                    weights[model] *= 0.7
        
        return weights
    
    def _get_blend_method(self, race_metadata: Optional[Dict[str, Any]]) -> str:
        """Determine which blend method was used."""
        if race_metadata:
            return "adaptive"
        else:
            return "default"
    
    def set_model_weights(self, model_weights: Dict[str, float]):
        """
        Manually set model weights.
        
        Args:
            model_weights: Dictionary of model weights
        """
        for model, weight in model_weights.items():
            if model in self.all_models:
                self.default_weights[model] = weight
    
    def enable_alternative_models(self, enabled_models: List[str], 
                                total_alternative_weight: float = 0.3):
        """
        Enable alternative models with specified total weight.
        
        Args:
            enabled_models: List of alternative model names to enable
            total_alternative_weight: Total weight to distribute among alternative models
        """
        if not enabled_models:
            return
        
        # Reduce legacy model weights
        legacy_reduction = total_alternative_weight / sum(
            self.default_weights[model] for model in self.legacy_models
            if model in self.default_weights
        )
        
        for model in self.legacy_models:
            if model in self.default_weights:
                self.default_weights[model] *= (1 - legacy_reduction)
        
        # Distribute weight among alternative models
        weight_per_alt_model = total_alternative_weight / len(enabled_models)
        for model in enabled_models:
            if model in self.alternative_models:
                self.default_weights[model] = weight_per_alt_model
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.default_weights.copy()
    
    def validate_predictions(self, predictions: Dict[str, np.ndarray]) -> bool:
        """
        Validate that predictions are consistent.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            True if valid, False otherwise
        """
        if not predictions:
            return False
        
        # Check all predictions have same length
        lengths = [len(pred) for pred in predictions.values() if pred is not None]
        if not lengths or len(set(lengths)) > 1:
            return False
        
        # Check for NaN values
        for model, pred in predictions.items():
            if pred is not None and np.any(np.isnan(pred)):
                if self.verbose:
                    self.logger.warning(f"NaN values found in {model} predictions")
                return False
        
        return True