"""
Helper module for loading and saving horse race prediction models.
Provides a consistent interface for all scripts that need to work with models.
"""

import os
import time
import json
import joblib
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple
from datetime import datetime

# Import TensorFlow and Keras
import tensorflow as tf
from tf.keras.models import load_model

from utils.env_setup import AppConfig


class ModelManager:
    """
    Helper class for managing model loading and saving operations.
    Provides a unified interface for all model handling operations.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the model manager.

        Args:
            config_path: Path to the configuration file
        """
        self.config = AppConfig(config_path)
        self.model_dir = self.config._config.models.model_dir

    def get_model_path(self, model_name: str = 'hybrid') -> Path:
        """
        Get the base path for a model.

        Args:
            model_name: Name of the model

        Returns:
            Path object for the model directory
        """
        return Path(self.model_dir) / model_name

    def get_version_path(self, db_type: str, train_type: str = 'full', date: Optional[str] = None) -> str:
        """
        Generate a version string based on database type, training type, and date.

        Args:
            db_type: Database type (e.g., 'full', '2years')
            train_type: Training type ('full' or 'incremental')
            date: Optional date string (default: current date)

        Returns:
            Version string in the format "{db_type}_{train_type}_v{YYYYMMDD}"
        """
        if date is None:
            date = time.strftime('%Y%m%d')
        return f"{db_type}_{train_type}_v{date}"

    def parse_version_string(self, version: str) -> Dict[str, str]:
        """
        Parse a version string into its components.

        Args:
            version: Version string (e.g., 'full_incremental_v20250407')

        Returns:
            Dictionary with parsed components (db_type, train_type, date)
        """
        parts = version.split('_')

        # Handle different version formats
        if len(parts) >= 3 and parts[-2] == 'v':
            # Format: {db_type}_{train_type}_v{date}
            db_type = parts[0]
            train_type = parts[1]
            date = parts[-1]
        elif len(parts) >= 2 and parts[-2] == 'v':
            # Format: {db_type}_v{date} (legacy)
            db_type = parts[0]
            train_type = 'full'  # Default to full for legacy versions
            date = parts[-1]
        elif len(parts) >= 3:
            # Format: {db_type}_{train_type}_v{date} (without separator)
            db_type = parts[0]
            train_type = parts[1]
            date = parts[2].replace('v', '')
        else:
            # Unknown format, use default values
            db_type = 'unknown'
            train_type = 'unknown'
            date = 'unknown'

        return {
            'db_type': db_type,
            'train_type': train_type,
            'date': date
        }

    def get_latest_model_version(self,
                                model_name: str = 'hybrid',
                                db_type: Optional[str] = None,
                                train_type: Optional[str] = None) -> Optional[str]:
        """
        Get the latest model version for the specified criteria.

        Args:
            model_name: Name of the model
            db_type: Optional database type filter
            train_type: Optional training type filter ('full' or 'incremental')

        Returns:
            Latest version string or None if no versions found
        """
        model_path = self.get_model_path(model_name)

        # Find all version directories
        if not model_path.exists():
            return None

        versions = []
        for d in model_path.iterdir():
            if not d.is_dir():
                continue

            # Check if directory matches our version pattern
            if db_type and not d.name.startswith(f"{db_type}_"):
                continue

            # Parse version to check train_type
            if train_type:
                parsed = self.parse_version_string(d.name)
                if parsed['train_type'] != train_type:
                    continue

            versions.append(d.name)

        if not versions:
            return None

        # Sort by version (newest first)
        versions.sort(reverse=True)
        return versions[0]

    def get_latest_base_model(self) -> Optional[str]:
        """
        Get the latest base model version from config.

        Returns:
            Latest base model version or None if not set
        """
        try:
            if hasattr(self.config._config.models, 'latest_base_model'):
                return self.config._config.models.latest_base_model

            if hasattr(self.config._config.models, 'latest_full_model'):
                return self.config._config.models.latest_full_model

            return None
        except (AttributeError, KeyError):
            return None

    def get_latest_incremental_model(self) -> Optional[str]:
        """
        Get the latest incremental model version from config.

        Returns:
            Latest incremental model version or None if not set
        """
        try:
            if hasattr(self.config._config.models, 'latest_incremental_model'):
                return self.config._config.models.latest_incremental_model
            return None
        except (AttributeError, KeyError):
            return None

    def update_config_model_reference(self, version: str, reference_type: str = 'base') -> bool:
        """
        Update model reference fields in config.yaml.

        Args:
            version: Version string to set
            reference_type: Type of reference to update ('base', 'full', or 'incremental')

        Returns:
            Success status
        """
        try:
            config_path = "config.yaml"

            # Read existing config
            with open(config_path, 'r') as f:
                config_content = f.read()

            # Parse YAML
            config_data = yaml.safe_load(config_content)

            # Update or add the models section
            if 'models' not in config_data:
                config_data['models'] = {}

            # Update appropriate field based on reference_type
            if reference_type == 'base' or reference_type == 'full':
                config_data['models']['latest_base_model'] = version
                config_data['models']['latest_full_model'] = version
            elif reference_type == 'incremental':
                config_data['models']['latest_incremental_model'] = version
            else:
                print(f"Unknown reference_type: {reference_type}")
                return False

            # Write updated config
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)

            print(f"Updated config.yaml with {reference_type} model: {version}")
            return True
        except Exception as e:
            print(f"Error updating config.yaml: {str(e)}")
            return False

    def resolve_model_path(self,
                          model_path: Optional[str] = None,
                          model_name: str = 'hybrid',
                          version: Optional[str] = None,
                          use_latest_base: bool = False,
                          use_latest_incremental: bool = False,
                          db_type: Optional[str] = None,
                          train_type: Optional[str] = None) -> Path:
        """
        Resolve the path to a model using various options, with fallbacks.

        Args:
            model_path: Explicit path to model (highest priority)
            model_name: Name of the model (default: 'hybrid')
            version: Specific version to use
            use_latest_base: Whether to use the latest base model from config
            use_latest_incremental: Whether to use the latest incremental model from config
            db_type: Optional database type to filter versions
            train_type: Optional training type to filter versions

        Returns:
            Path object for the resolved model

        Raises:
            ValueError: If no valid model path could be resolved
        """
        # Priority 1: Explicit model_path
        if model_path:
            return Path(model_path)

        # Priority 2: Use latest base or incremental model from config
        if use_latest_base:
            latest_base = self.get_latest_base_model()
            if latest_base:
                return self.get_model_path(model_name) / latest_base

        if use_latest_incremental:
            latest_incremental = self.get_latest_incremental_model()
            if latest_incremental:
                return self.get_model_path(model_name) / latest_incremental

        # Priority 3: Specific version with specified model name
        if version:
            return self.get_model_path(model_name) / version

        # Priority 4: Latest version with optional db_type and train_type filters
        latest_version = self.get_latest_model_version(model_name, db_type, train_type)
        if latest_version:
            return self.get_model_path(model_name) / latest_version

        # No valid path could be resolved
        raise ValueError("Could not resolve a valid model path. Please specify a model_path, version, or ensure latest model references are set in config.")

    def save_model(self,
                  model: Any,
                  model_path: Path,
                  metadata: Optional[Dict] = None,
                  is_rf: bool = True,
                  train_type: str = 'full',
                  db_type: Optional[str] = None,
                  update_config: bool = True) -> Path:
        """
        Save a model to disk with optional metadata.

        Args:
            model: Model object to save
            model_path: Path where to save the model
            metadata: Optional metadata to save with the model
            is_rf: Whether this is a Random Forest model
            train_type: Training type ('full' or 'incremental')
            db_type: Database type used for training
            update_config: Whether to update config with latest model reference

        Returns:
            Path where the model was saved
        """
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        if metadata:
            save_data = {
                'model': model,
                'metadata': metadata,
                'train_type': train_type,
                'db_type': db_type,
                'timestamp': datetime.now().isoformat()
            }
        else:
            save_data = model

        # Save the model
        if is_rf:
            # For RF models, try to use CalibratedRegressor's save method if available
            if hasattr(model, 'save') and callable(getattr(model, 'save')):
                model.save(model_path)
                print(f"Saved model using model's own save method to: {model_path}")
            else:
                # Fallback to joblib dump
                joblib.dump(save_data, model_path)
                print(f"Saved model with joblib to: {model_path}")
        else:
            # For non-RF models (e.g., LSTM), use the model's save method
            if hasattr(model, 'save') and callable(getattr(model, 'save')):
                model.save(model_path)
                print(f"Saved non-RF model to: {model_path}")
            else:
                # Fallback to joblib dump
                joblib.dump(save_data, model_path)
                print(f"Saved non-RF model with joblib to: {model_path}")

        # Update config if requested
        if update_config and db_type:
            version = model_path.parent.name
            # Update the appropriate reference based on train_type
            if train_type == 'full':
                self.update_config_model_reference(version, 'full')
            elif train_type == 'incremental':
                self.update_config_model_reference(version, 'incremental')

        return model_path

    def load_model(self,
                  model_path: Path,
                  is_rf: bool = True,
                  verbose: bool = False) -> Tuple[Any, Optional[Dict]]:
        """
        Load a model from disk.

        Args:
            model_path: Path to the model
            is_rf: Whether this is a Random Forest model
            verbose: Whether to print verbose output

        Returns:
            Tuple of (model, metadata) where metadata may be None

        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        metadata = None

        try:
            if is_rf:
                # For RF models, try CalibratedRegressor.load first
                try:
                    from model_training.regressions.isotonic_calibration import CalibratedRegressor
                    model = CalibratedRegressor.load(model_path)
                    if verbose:
                        print(f"Loaded RF model with CalibratedRegressor from {model_path}")
                except Exception as e:
                    if verbose:
                        print(f"CalibratedRegressor.load failed: {str(e)}")

                    # Fallback to joblib load
                    loaded_data = joblib.load(model_path)

                    # Handle different saving formats
                    if isinstance(loaded_data, dict) and 'model' in loaded_data:
                        model = loaded_data['model']
                        metadata = loaded_data.get('metadata')
                        if verbose:
                            print(f"Loaded model from dictionary with keys: {list(loaded_data.keys())}")
                    else:
                        # Direct model object
                        model = loaded_data
                        if verbose:
                            print(f"Loaded model directly from {model_path}")
            else:
                # For non-RF models (e.g., LSTM), use appropriate method
                if str(model_path).endswith('.h5') or model_path.is_dir():
                    # Keras/TensorFlow model
                    model = load_model(model_path)
                    if verbose:
                        print(f"Loaded Keras/TensorFlow model from {model_path}")
                else:
                    # Default to joblib for other types
                    loaded_data = joblib.load(model_path)

                    # Check if it's a dict with model key
                    if isinstance(loaded_data, dict) and 'model' in loaded_data:
                        model = loaded_data['model']
                        metadata = loaded_data.get('metadata')
                    else:
                        model = loaded_data

                    if verbose:
                        print(f"Loaded non-RF model from {model_path}")

            return model, metadata

        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def save_model_artifacts(self,
                            base_path: Path,
                            rf_model=None,
                            lstm_model=None,
                            orchestrator_state=None,
                            history=None,
                            model_config=None,
                            db_type=None,
                            train_type='full',
                            update_config=True) -> Dict[str, Path]:
        """
        Save all model artifacts to disk.

        Args:
            base_path: Base path for saving artifacts
            rf_model: Random Forest model (optional)
            lstm_model: LSTM model (optional)
            orchestrator_state: Feature engineering state (optional)
            history: Training history (optional)
            model_config: Model configuration (optional)
            db_type: Database type used for training
            train_type: Training type ('full' or 'incremental')
            update_config: Whether to update config with latest model references

        Returns:
            Dictionary of paths where artifacts were saved
        """
        saved_paths = {}

        # Create version directory if needed
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True)

        # Save RF model
        if rf_model is not None:
            rf_path = base_path / "hybrid_rf_model.joblib"
            self.save_model(
                rf_model, rf_path,
                is_rf=True,
                train_type=train_type,
                db_type=db_type,
                update_config=update_config
            )
            saved_paths['rf_model'] = rf_path

        # Save LSTM model
        if lstm_model is not None:
            lstm_path = base_path / "hybrid_lstm_model"
            if hasattr(lstm_model, 'save'):
                lstm_model.save(lstm_path)
                saved_paths['lstm_model'] = lstm_path

        # Save LSTM training history
        if history is not None:
            history_path = base_path / 'lstm_history.joblib'
            joblib.dump(history, history_path)
            saved_paths['history'] = history_path

        # Save orchestrator state
        if orchestrator_state is not None:
            feature_path = base_path / "hybrid_feature_engineer.joblib"
            joblib.dump(orchestrator_state, feature_path)
            saved_paths['feature_engineer'] = feature_path

        # Update model_config with training type info if not present
        if model_config is not None and 'train_type' not in model_config:
            model_config['train_type'] = train_type

        # Save model configuration
        if model_config is not None:
            config_path = base_path / 'model_config.json'
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2, default=str)
            saved_paths['model_config'] = config_path

        return saved_paths

    def load_model_artifacts(self,
                           base_path: Path,
                           load_rf: bool = True,
                           load_lstm: bool = True,
                           load_feature_config: bool = True,
                           verbose: bool = False) -> Dict[str, Any]:
        """
        Load all model artifacts from disk.

        Args:
            base_path: Base path for loading artifacts
            load_rf: Whether to load RF model
            load_lstm: Whether to load LSTM model
            load_feature_config: Whether to load feature configuration
            verbose: Whether to print verbose output

        Returns:
            Dictionary of loaded artifacts
        """
        artifacts = {}

        # Load RF model
        if load_rf:
            rf_path = base_path / "hybrid_rf_model.joblib"
            if rf_path.exists():
                try:
                    rf_model, rf_metadata = self.load_model(rf_path, is_rf=True, verbose=verbose)
                    artifacts['rf_model'] = rf_model
                    if rf_metadata:
                        artifacts['rf_metadata'] = rf_metadata
                    if verbose:
                        print(f"Loaded RF model from {rf_path}")
                except Exception as e:
                    print(f"Error loading RF model: {str(e)}")

        # Load LSTM model
        if load_lstm:
            lstm_path = base_path / "hybrid_lstm_model"
            if lstm_path.exists():
                try:
                    lstm_model = load_model(lstm_path)
                    artifacts['lstm_model'] = lstm_model
                    if verbose:
                        print(f"Loaded LSTM model from {lstm_path}")

                    # Try to load history
                    history_path = base_path / 'lstm_history.joblib'
                    if history_path.exists():
                        history = joblib.load(history_path)
                        artifacts['history'] = history
                except Exception as e:
                    print(f"Error loading LSTM model: {str(e)}")

        # Load feature configuration
        if load_feature_config:
            feature_path = base_path / "hybrid_feature_engineer.joblib"
            if feature_path.exists():
                try:
                    feature_config = joblib.load(feature_path)
                    artifacts['feature_config'] = feature_config
                    if verbose:
                        print(f"Loaded feature configuration from {feature_path}")
                except Exception as e:
                    print(f"Error loading feature configuration: {str(e)}")

        # Load model configuration
        config_path = base_path / 'model_config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                artifacts['model_config'] = model_config
                if verbose:
                    print(f"Loaded model configuration from {config_path}")
            except Exception as e:
                print(f"Error loading model configuration: {str(e)}")

        return artifacts


# Create a singleton instance for easy import
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the singleton model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager