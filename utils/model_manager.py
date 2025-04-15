import os
import json
import joblib
import yaml
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model

from utils.env_setup import AppConfig


class ModelManager:
    """Helper class for managing model operations."""

    def __init__(self):
        """Initialize the model manager."""
        self.config = AppConfig()
        self.model_dir = self.config._config.models.model_dir

    def get_model_path(self, model_type ='hybrid', db_type=None):
        """Get the base path for a model."""
        if db_type is None:
            db_type = self.config._config.base.active_db
        return Path(self.model_dir) / db_type / model_type

    def get_version_path(self, db_type, train_type='full'):
        """Generate a version string with the current date."""
        date = datetime.now().strftime('%Y%m%d')
        return f"{db_type}_{train_type}_v{date}"

    def update_config_reference(self, version, model_type='hybrid'):
        """Update model reference in config.yaml."""
        try:
            with open("config.yaml", 'r') as f:
                config_data = yaml.safe_load(f)

            if 'models' not in config_data:
                config_data['models'] = {}

            config_data['models'][f"latest_{model_type}_model"] = version

            # Set appropriate base or incremental reference
            if '_incremental_' in version:
                config_data['models']['latest_incremental_model'] = version
            else:
                config_data['models']['latest_base_model'] = version
                config_data['models']['latest_full_model'] = version

            with open("config.yaml", 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)

            print(f"Updated config with model reference: {version}")
            return True
        except Exception as e:
            print(f"Error updating config: {e}")
            return False

    def save_model_artifacts(self, base_path, rf_model=None, lstm_model=None,
                            orchestrator_state=None, history=None, model_config=None,
                            db_type=None, train_type='full'):
        """Save model artifacts to disk."""
        # Create directory if needed
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save RF model
        if rf_model is not None:
            rf_path = base_path / "hybrid_rf_model.joblib"
            joblib.dump(rf_model, rf_path)
            saved_paths['rf_model'] = rf_path
            print(f"Saved RF model to: {rf_path}")

        # Save LSTM model
        if lstm_model is not None:
            lstm_path = base_path / "hybrid_lstm_model.keras"
            lstm_model.save(lstm_path)
            saved_paths['lstm_model'] = lstm_path
            print(f"Saved LSTM model to: {lstm_path}")

        # Save feature engineering state
        if orchestrator_state is not None:
            feature_path = base_path / "hybrid_feature_engineer.joblib"
            joblib.dump(orchestrator_state, feature_path)
            saved_paths['feature_engineer'] = feature_path
            print(f"Saved feature state to: {feature_path}")

        # Save history
        if history is not None:
            history_path = base_path / "lstm_history.joblib"
            joblib.dump(history, history_path)
            saved_paths['history'] = history_path
            print(f"Saved history to: {history_path}")

        # Save config
        if model_config is not None:
            # Add version and train_type if not present
            if 'version' not in model_config:
                model_config['version'] = base_path.name
            if 'train_type' not in model_config:
                model_config['train_type'] = train_type

            config_path = base_path / "model_config.json"
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            saved_paths['model_config'] = config_path
            print(f"Saved config to: {config_path}")

        # Update config reference
        version = base_path.name
        self.update_config_reference(version)

        return saved_paths

    def load_model_artifacts(self, base_path, load_rf=True, load_lstm=True, load_feature_config=True):
        """Load model artifacts from disk."""
        base_path = Path(base_path)
        artifacts = {}

        # Load RF model
        if load_rf:
            rf_path = base_path / "hybrid_rf_model.joblib"
            if rf_path.exists():
                try:
                    from model_training.regressions.isotonic_calibration import CalibratedRegressor
                    artifacts['rf_model'] = CalibratedRegressor.load(rf_path)
                except:
                    artifacts['rf_model'] = joblib.load(rf_path)
                print(f"Loaded RF model from: {rf_path}")

        # Load LSTM model
        if load_lstm:
            lstm_path = base_path / "hybrid_lstm_model.keras"
            if lstm_path.exists():
                artifacts['lstm_model'] = load_model(lstm_path)
                print(f"Loaded LSTM model from: {lstm_path}")

                # Load history if available
                history_path = base_path / "lstm_history.joblib"
                if history_path.exists():
                    artifacts['history'] = joblib.load(history_path)

        # Load feature config
        if load_feature_config:
            feature_path = base_path / "hybrid_feature_engineer.joblib"
            if feature_path.exists():
                artifacts['feature_config'] = joblib.load(feature_path)
                print(f"Loaded feature config from: {feature_path}")

        # Load model config
        config_path = base_path / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                artifacts['model_config'] = json.load(f)

        return artifacts


# Singleton instance
_model_manager = None

def get_model_manager():
    """Get the singleton model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager