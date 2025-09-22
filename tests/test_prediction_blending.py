"""
Tests for the PredictionBlender class.

This test suite validates:
- Configuration loading and validation
- Weight selection based on race metadata
- Fail-fast behavior for invalid inputs
- Blending calculation correctness
"""

import pytest
import numpy as np
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from race_prediction.prediction_blender import PredictionBlender


class TestPredictionBlender:
    """Test suite for PredictionBlender class."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid configuration for testing."""
        return {
            'model_blending': {
                'default': {
                    'rf': 0.5,
                    'lstm': 0.3,
                    'tabnet': 0.2
                },
                'rules': {
                    'sprint': {
                        'condition': {
                            'typec': ['P'],
                            'distance_min': 800,
                            'distance_max': 1400
                        },
                        'weights': {
                            'rf': 0.6,
                            'lstm': 0.2,
                            'tabnet': 0.2
                        }
                    },
                    'long_distance': {
                        'condition': {
                            'typec': ['P'],
                            'distance_min': 2000,
                            'distance_max': 4000
                        },
                        'weights': {
                            'rf': 0.4,
                            'lstm': 0.5,
                            'tabnet': 0.1
                        }
                    },
                    'large_field': {
                        'condition': {
                            'partant_min': 12
                        },
                        'weights': {
                            'rf': 0.4,
                            'lstm': 0.3,
                            'tabnet': 0.3
                        }
                    }
                }
            }
        }

    @pytest.fixture
    def temp_config_file(self, valid_config):
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            return f.name

    @pytest.fixture
    def sample_predictions(self):
        """Sample predictions for testing."""
        return {
            'rf': np.array([1.2, 2.1, 3.5, 4.0, 5.3]),
            'lstm': np.array([1.1, 2.3, 3.2, 4.1, 5.1]),
            'tabnet': np.array([1.3, 2.0, 3.6, 3.9, 5.4])
        }

    def test_init_with_valid_config(self, temp_config_file):
        """Test initialization with valid configuration."""
        blender = PredictionBlender(temp_config_file)
        
        assert blender.config_path == Path(temp_config_file)
        assert 'default' in blender.blending_config
        assert 'rules' in blender.blending_config
        assert blender.required_models == {'rf', 'lstm', 'tabnet'}
        
        # Cleanup
        os.unlink(temp_config_file)

    def test_init_missing_config_file(self):
        """Test initialization fails with missing config file."""
        with pytest.raises(FileNotFoundError):
            PredictionBlender('/nonexistent/path/config.yaml')

    def test_init_invalid_yaml(self):
        """Test initialization fails with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
            
        try:
            with pytest.raises(ValueError, match="Invalid YAML configuration"):
                PredictionBlender(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_validation_missing_model_blending(self):
        """Test validation fails when model_blending section is missing."""
        config = {'other_section': {}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name
            
        try:
            with pytest.raises(ValueError, match="must contain 'model_blending' section"):
                PredictionBlender(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_validation_missing_default(self, valid_config):
        """Test validation fails when default weights are missing."""
        del valid_config['model_blending']['default']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            temp_path = f.name
            
        try:
            with pytest.raises(ValueError, match="must contain 'default' weights"):
                PredictionBlender(temp_path)
        finally:
            os.unlink(temp_path)

    def test_weight_validation_missing_models(self, valid_config):
        """Test validation fails when model weights are missing."""
        del valid_config['model_blending']['default']['lstm']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            temp_path = f.name
            
        try:
            with pytest.raises(ValueError, match="Missing model weights.*lstm"):
                PredictionBlender(temp_path)
        finally:
            os.unlink(temp_path)

    def test_weight_validation_weights_dont_sum_to_one(self, valid_config):
        """Test validation fails when weights don't sum to 1.0."""
        valid_config['model_blending']['default']['rf'] = 0.8  # Now sums to 1.3
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            temp_path = f.name
            
        try:
            with pytest.raises(ValueError, match="must sum to 1.0"):
                PredictionBlender(temp_path)
        finally:
            os.unlink(temp_path)

    def test_weight_validation_negative_weight(self, valid_config):
        """Test validation fails with negative weights."""
        valid_config['model_blending']['default']['rf'] = -0.1
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            temp_path = f.name
            
        try:
            with pytest.raises(ValueError, match="must be positive number"):
                PredictionBlender(temp_path)
        finally:
            os.unlink(temp_path)

    def test_get_blend_weights_default(self, temp_config_file):
        """Test getting default weights when no rules match."""
        blender = PredictionBlender(temp_config_file)
        
        race_metadata = {
            'typec': 'T',  # Trot - no matching rule
            'distance': 1600,
            'partant': 8
        }
        
        weights = blender.get_blend_weights(race_metadata)
        
        assert weights['rf'] == 0.5
        assert weights['lstm'] == 0.3
        assert weights['tabnet'] == 0.2
        assert weights['_applied_rule'] == 'default'
        
        # Cleanup
        os.unlink(temp_config_file)

    def test_get_blend_weights_sprint_rule(self, temp_config_file):
        """Test getting sprint rule weights."""
        blender = PredictionBlender(temp_config_file)
        
        race_metadata = {
            'typec': 'P',
            'distance': 1200,  # Sprint distance
            'partant': 8
        }
        
        weights = blender.get_blend_weights(race_metadata)
        
        assert weights['rf'] == 0.6
        assert weights['lstm'] == 0.2
        assert weights['tabnet'] == 0.2
        assert weights['_applied_rule'] == 'sprint'
        
        # Cleanup
        os.unlink(temp_config_file)

    def test_get_blend_weights_long_distance_rule(self, temp_config_file):
        """Test getting long distance rule weights."""
        blender = PredictionBlender(temp_config_file)
        
        race_metadata = {
            'typec': 'P',
            'distance': 2400,  # Long distance
            'partant': 10
        }
        
        weights = blender.get_blend_weights(race_metadata)
        
        assert weights['rf'] == 0.4
        assert weights['lstm'] == 0.5
        assert weights['tabnet'] == 0.1
        assert weights['_applied_rule'] == 'long_distance'
        
        # Cleanup
        os.unlink(temp_config_file)

    def test_get_blend_weights_large_field_rule(self, temp_config_file):
        """Test getting large field rule weights."""
        blender = PredictionBlender(temp_config_file)
        
        race_metadata = {
            'typec': 'P',
            'distance': 1600,
            'partant': 15  # Large field
        }
        
        weights = blender.get_blend_weights(race_metadata)
        
        assert weights['rf'] == 0.4
        assert weights['lstm'] == 0.3
        assert weights['tabnet'] == 0.3
        assert weights['_applied_rule'] == 'large_field'
        
        # Cleanup
        os.unlink(temp_config_file)

    def test_get_blend_weights_invalid_metadata(self, temp_config_file):
        """Test fail-fast with invalid metadata."""
        blender = PredictionBlender(temp_config_file)
        
        with pytest.raises(ValueError, match="race_metadata must be dict"):
            blender.get_blend_weights("invalid")
            
        # Cleanup
        os.unlink(temp_config_file)

    def test_matches_condition_exact_match(self, temp_config_file):
        """Test condition matching with exact values."""
        blender = PredictionBlender(temp_config_file)
        
        race_metadata = {'typec': 'P', 'distance': 1200}
        condition = {'typec': ['P'], 'distance': 1200}
        
        assert blender._matches_condition(race_metadata, condition)
        
        # Test mismatch
        condition = {'typec': ['T'], 'distance': 1200}
        assert not blender._matches_condition(race_metadata, condition)
        
        # Cleanup
        os.unlink(temp_config_file)

    def test_matches_condition_min_max(self, temp_config_file):
        """Test condition matching with min/max values."""
        blender = PredictionBlender(temp_config_file)
        
        race_metadata = {'distance': 1200, 'partant': 10}
        
        # Test within range
        condition = {'distance_min': 800, 'distance_max': 1400}
        assert blender._matches_condition(race_metadata, condition)
        
        # Test below range
        condition = {'distance_min': 1500, 'distance_max': 2000}
        assert not blender._matches_condition(race_metadata, condition)
        
        # Test above range
        condition = {'distance_min': 800, 'distance_max': 1100}
        assert not blender._matches_condition(race_metadata, condition)
        
        # Test min only
        condition = {'partant_min': 8}
        assert blender._matches_condition(race_metadata, condition)
        
        condition = {'partant_min': 12}
        assert not blender._matches_condition(race_metadata, condition)
        
        # Cleanup
        os.unlink(temp_config_file)

    def test_blend_predictions_success(self, temp_config_file, sample_predictions):
        """Test successful prediction blending."""
        blender = PredictionBlender(temp_config_file)
        
        race_metadata = {'typec': 'P', 'distance': 1200}  # Should match sprint rule
        
        result = blender.blend_predictions(sample_predictions, race_metadata)
        
        assert 'blended_predictions' in result
        assert 'weights_used' in result
        assert 'applied_rule' in result
        assert 'individual_predictions' in result
        
        # Verify blending calculation
        expected_blend = (
            sample_predictions['rf'] * 0.6 +
            sample_predictions['lstm'] * 0.2 +
            sample_predictions['tabnet'] * 0.2
        )
        
        np.testing.assert_array_almost_equal(
            result['blended_predictions'], expected_blend, decimal=6
        )
        
        assert result['applied_rule'] == 'sprint'
        assert result['weights_used']['rf'] == 0.6
        assert result['weights_used']['lstm'] == 0.2
        assert result['weights_used']['tabnet'] == 0.2
        
        # Cleanup
        os.unlink(temp_config_file)

    def test_validate_predictions_missing_models(self, temp_config_file):
        """Test validation fails with missing model predictions."""
        blender = PredictionBlender(temp_config_file)
        
        incomplete_predictions = {
            'rf': np.array([1, 2, 3]),
            'lstm': np.array([1, 2, 3])
            # Missing tabnet
        }
        
        with pytest.raises(ValueError, match="Missing model predictions.*tabnet"):
            blender._validate_predictions(incomplete_predictions)
            
        # Cleanup
        os.unlink(temp_config_file)

    def test_validate_predictions_misaligned_lengths(self, temp_config_file):
        """Test validation fails with misaligned prediction lengths."""
        blender = PredictionBlender(temp_config_file)
        
        misaligned_predictions = {
            'rf': np.array([1, 2, 3]),
            'lstm': np.array([1, 2]),  # Different length
            'tabnet': np.array([1, 2, 3])
        }
        
        with pytest.raises(ValueError, match="Prediction lengths misaligned"):
            blender._validate_predictions(misaligned_predictions)
            
        # Cleanup
        os.unlink(temp_config_file)

    def test_validate_predictions_none_values(self, temp_config_file):
        """Test validation fails with None predictions."""
        blender = PredictionBlender(temp_config_file)
        
        none_predictions = {
            'rf': np.array([1, 2, 3]),
            'lstm': None,
            'tabnet': np.array([1, 2, 3])
        }
        
        with pytest.raises(ValueError, match="Predictions for lstm cannot be None"):
            blender._validate_predictions(none_predictions)
            
        # Cleanup
        os.unlink(temp_config_file)

    def test_validate_predictions_empty_arrays(self, temp_config_file):
        """Test validation fails with empty prediction arrays."""
        blender = PredictionBlender(temp_config_file)
        
        empty_predictions = {
            'rf': np.array([1, 2, 3]),
            'lstm': np.array([]),  # Empty array
            'tabnet': np.array([1, 2, 3])
        }
        
        with pytest.raises(ValueError, match="Predictions for lstm cannot be empty"):
            blender._validate_predictions(empty_predictions)
            
        # Cleanup
        os.unlink(temp_config_file)

    def test_get_available_rules(self, temp_config_file):
        """Test getting list of available rules."""
        blender = PredictionBlender(temp_config_file)
        
        rules = blender.get_available_rules()
        
        expected_rules = ['sprint', 'long_distance', 'large_field']
        assert set(rules) == set(expected_rules)
        
        # Cleanup
        os.unlink(temp_config_file)

    def test_describe_rule(self, temp_config_file):
        """Test getting rule description."""
        blender = PredictionBlender(temp_config_file)
        
        rule_desc = blender.describe_rule('sprint')
        
        expected = {
            'condition': {
                'typec': ['P'],
                'distance_min': 800,
                'distance_max': 1400
            },
            'weights': {
                'rf': 0.6,
                'lstm': 0.2,
                'tabnet': 0.2
            }
        }
        
        assert rule_desc == expected
        
        # Test nonexistent rule
        with pytest.raises(ValueError, match="Rule 'nonexistent' not found"):
            blender.describe_rule('nonexistent')
            
        # Cleanup
        os.unlink(temp_config_file)

    def test_multiple_rule_matching_priority(self, temp_config_file):
        """Test that first matching rule is used when multiple rules could match."""
        blender = PredictionBlender(temp_config_file)
        
        # This metadata could match both sprint and large_field rules
        race_metadata = {
            'typec': 'P',
            'distance': 1200,  # Matches sprint
            'partant': 15      # Matches large_field
        }
        
        weights = blender.get_blend_weights(race_metadata)
        
        # Should use the first matching rule (sprint, as it appears first in config)
        assert weights['_applied_rule'] == 'sprint'
        assert weights['rf'] == 0.6
        
        # Cleanup
        os.unlink(temp_config_file)

    def test_blending_calculation_correctness(self, temp_config_file):
        """Test that blending calculations are mathematically correct."""
        blender = PredictionBlender(temp_config_file)
        
        # Create known predictions
        predictions = {
            'rf': np.array([2.0, 4.0, 6.0]),
            'lstm': np.array([1.0, 3.0, 5.0]),
            'tabnet': np.array([3.0, 5.0, 7.0])
        }
        
        race_metadata = {'typec': 'no_match'}  # Use default weights (0.5, 0.3, 0.2)
        
        result = blender.blend_predictions(predictions, race_metadata)
        
        # Calculate expected result manually
        expected = (
            predictions['rf'] * 0.5 +     # [1.0, 2.0, 3.0]
            predictions['lstm'] * 0.3 +   # [0.3, 0.9, 1.5]  
            predictions['tabnet'] * 0.2   # [0.6, 1.0, 1.4]
        )  # Total: [1.9, 3.9, 5.9]
        
        np.testing.assert_array_almost_equal(
            result['blended_predictions'], 
            expected, 
            decimal=10
        )
        
        # Cleanup
        os.unlink(temp_config_file)


def test_prediction_blender_integration():
    """Integration test using actual config.yaml structure."""
    # Create a minimal config that matches the actual structure
    config = {
        'model_blending': {
            'default': {
                'rf': 0.5,
                'lstm': 0.3,
                'tabnet': 0.2
            },
            'rules': {
                'sprint': {
                    'condition': {
                        'typec': ['P'],
                        'distance_min': 800,
                        'distance_max': 1400
                    },
                    'weights': {
                        'rf': 0.6,
                        'lstm': 0.2,
                        'tabnet': 0.2
                    }
                }
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    
    try:
        blender = PredictionBlender(temp_path)
        
        # Test sprint race
        sprint_metadata = {'typec': 'P', 'distance': 1200, 'partant': 10}
        sprint_predictions = {
            'rf': np.array([1.5, 2.5, 3.5]),
            'lstm': np.array([1.4, 2.6, 3.4]),
            'tabnet': np.array([1.6, 2.4, 3.6])
        }
        
        result = blender.blend_predictions(sprint_predictions, sprint_metadata)
        
        # Should use sprint rule
        assert result['applied_rule'] == 'sprint'
        assert result['weights_used']['rf'] == 0.6
        
        # Test non-matching race (should use default)
        default_metadata = {'typec': 'T', 'distance': 2000, 'partant': 8}
        
        result_default = blender.blend_predictions(sprint_predictions, default_metadata)
        
        assert result_default['applied_rule'] == 'default'
        assert result_default['weights_used']['rf'] == 0.5
        
    finally:
        os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__])