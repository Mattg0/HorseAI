import json
import joblib
import yaml
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model

from utils.env_setup import AppConfig

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False


class ModelManager:
    """Simple model manager for saving and loading models."""

    def __init__(self):
        """Initialize the model manager."""
        self.config = AppConfig()
        self.model_dir = Path(self.config._config.models.model_dir)

    def get_model_path(self):
        """Get the path of the latest model from config."""
        # Read latest model from config
        try:
            with open("config.yaml", 'r') as f:
                config_data = yaml.safe_load(f)

            if 'models' in config_data and 'latest_model' in config_data['models']:
                latest_model = config_data['models']['latest_model']
                model_path = self.model_dir / latest_model

                if model_path.exists():
                    return model_path
        except:
            pass

        # Fallback: find the most recent model directory
        latest_dir = self._find_latest_model_dir()
        if latest_dir is None:
            print("No existing models found. New models will be created during training.")
            return None

        return latest_dir

    def _find_latest_model_dir(self):
        """Find the most recent model directory by scanning the filesystem."""
        if not self.model_dir.exists():
            print(f"Model directory {self.model_dir} does not exist yet")
            return None

        # Look for date directories
        date_dirs = [d for d in self.model_dir.iterdir()
                     if d.is_dir() and len(d.name) == 10 and '-' in d.name]

        if not date_dirs:
            print(f"No model directories found in {self.model_dir}")
            return None

        # Sort by date and get the latest
        latest_date_dir = sorted(date_dirs)[-1]

        # Find the latest timestamp directory
        timestamp_dirs = [d for d in latest_date_dir.iterdir() if d.is_dir()]

        if not timestamp_dirs:
            print(f"No model directories found in {latest_date_dir}")
            return None

        return sorted(timestamp_dirs)[-1]

    def save_models(self, rf_model=None, lstm_model=None, tabnet_model=None, tabnet_scaler=None, 
                    tabnet_feature_columns=None, feature_state=None):
        """Save models with simple date/db based path."""
        # Ensure models directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Get current date and db type
        date_str = datetime.now().strftime('%Y-%m-%d')
        timestamp_str = datetime.now().strftime('%H%M%S')
        db_type = self.config._config.base.active_db

        # Create path: models/YYYY-MM-DD/db_HHMMSS
        save_path = self.model_dir / date_str / f"{db_type}_{timestamp_str}"
        save_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save RF model
        if rf_model is not None:
            rf_path = save_path / "hybrid_rf_model.joblib"
            joblib.dump(rf_model, rf_path)
            saved_files['rf_model'] = rf_path

        # Save LSTM model
        if lstm_model is not None:
            lstm_path = save_path / "hybrid_lstm_model.keras"
            lstm_model.save(lstm_path)
            saved_files['lstm_model'] = lstm_path

        # Save TabNet model and related files
        if tabnet_model is not None:
            try:
                # Save TabNet model
                tabnet_path = save_path / "tabnet_model"
                tabnet_model.save_model(str(tabnet_path))
                saved_files['tabnet_model'] = tabnet_path
                
                # Save TabNet scaler if available
                if tabnet_scaler is not None:
                    scaler_path = save_path / "tabnet_scaler.joblib"
                    joblib.dump(tabnet_scaler, scaler_path)
                    saved_files['tabnet_scaler'] = scaler_path
                
                # Save TabNet feature configuration
                if tabnet_feature_columns is not None:
                    config_path = save_path / "tabnet_config.json"
                    tabnet_config = {
                        'feature_columns': tabnet_feature_columns,
                        'created_at': datetime.now().isoformat()
                    }
                    with open(config_path, 'w') as f:
                        json.dump(tabnet_config, f, indent=2)
                    saved_files['tabnet_config'] = config_path
                    
            except Exception as e:
                print(f"Warning: Could not save TabNet model: {e}")

        # Save feature engineering state if provided
        if feature_state is not None:
            feature_path = save_path / "hybrid_feature_engineer.joblib"
            joblib.dump(feature_state, feature_path)
            saved_files['feature_engineer'] = feature_path

        # Save minimal config
        config_data = {
            'db_type': db_type,
            'created_at': datetime.now().isoformat(),
            'blending_approach': 'prediction_time_configurable'
        }

        config_path = save_path / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        saved_files['model_config'] = config_path

        # Update config.yaml with latest model
        relative_path = save_path.relative_to(self.model_dir)
        self._update_config(str(relative_path))

        print(f"Models saved to: {save_path}")
        return saved_files

    def load_models(self, model_path=None):
        """Load models from path."""
        if model_path is None:
            model_path = self.get_model_path()
        else:
            model_path = Path(model_path)

        if model_path is None:
            raise ValueError("No model path available for loading")

        models = {}

        # Load RF model
        rf_path = model_path / "hybrid_rf_model.joblib"
        if rf_path.exists():
            models['rf_model'] = joblib.load(rf_path)

        # Load LSTM model
        lstm_path = model_path / "hybrid_lstm_model.keras"
        if lstm_path.exists():
            models['lstm_model'] = load_model(lstm_path)

        # Load TabNet model and related files
        if TABNET_AVAILABLE:
            tabnet_path = model_path / "tabnet_model.zip"
            scaler_path = model_path / "tabnet_scaler.joblib"
            config_path = model_path / "tabnet_config.json"


            if tabnet_path.exists():
                try:
                    tabnet_model = TabNetRegressor()
                    tabnet_model.load_model(str(tabnet_path))
                    models['tabnet_model'] = tabnet_model

                    # Load TabNet scaler if available
                    if scaler_path.exists():
                        models['tabnet_scaler'] = joblib.load(scaler_path)
                    
                    # Load TabNet configuration if available
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            models['tabnet_config'] = json.load(f)
                            models['tabnet_feature_columns'] = models['tabnet_config'].get('feature_columns', [])
                            
                except Exception as e:
                    print(f"Warning: Could not load TabNet model: {e}")

        # Load feature engineering state
        feature_path = model_path / "hybrid_feature_engineer.joblib"
        if feature_path.exists():
            models['feature_state'] = joblib.load(feature_path)

        # Load model config
        config_path = model_path / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                models['model_config'] = json.load(f)

        return models

    def _update_config(self, model_path):
        """Update config.yaml with latest model."""
        try:
            with open("config.yaml", 'r') as f:
                config_data = yaml.safe_load(f)

            if 'models' not in config_data:
                config_data['models'] = {}

            config_data['models']['latest_model'] = model_path

            with open("config.yaml", 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)

        except Exception as e:
            print(f"Warning: Could not update config: {e}")


# Singleton instance
_model_manager = None


def get_model_manager():
    """Get the singleton model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager