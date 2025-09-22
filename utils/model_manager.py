import json
import joblib
import yaml
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model

from utils.env_setup import AppConfig


class ModelManager:
    """Simple model manager for saving and loading models."""

    def __init__(self):
        """Initialize the model manager."""
        self.config = AppConfig()
        self.model_dir = Path(self.config._config.models.model_dir)

    def get_model_path(self):
        """Get the path of the latest RF model from config."""
        return self.get_model_path_by_type('rf')
        
    def get_model_path_by_type(self, model_type='rf'):
        """Get the path of the latest model by type (rf, tabnet, feedforward)."""
        try:
            with open("config.yaml", 'r') as f:
                config_data = yaml.safe_load(f)

            # Check new format first (latest_models with rf, tabnet entries)
            if 'models' in config_data and 'latest_models' in config_data['models']:
                latest_models = config_data['models']['latest_models']
                if model_type in latest_models and latest_models[model_type]:
                    model_path = self.model_dir / latest_models[model_type]
                    if model_path.exists():
                        return model_path

            # Fallback to old format for legacy models (backward compatibility)
            if model_type in ['legacy', 'rf'] and 'models' in config_data and 'latest_model' in config_data['models']:
                latest_model = config_data['models']['latest_model']
                model_path = self.model_dir / latest_model
                if model_path.exists():
                    return model_path
        except Exception as e:
            print(f"Error reading config for model type {model_type}: {e}")

        # Fallback: find the most recent model directory (for RF/legacy models)
        if model_type in ['legacy', 'rf']:
            latest_dir = self._find_latest_model_dir()
            if latest_dir is None:
                print("No existing models found. New models will be created during training.")
                return None
            return latest_dir

        return None
        
    def get_all_model_paths(self):
        """Get paths for all available model types."""
        paths = {}
        for model_type in ['rf', 'tabnet']:
            path = self.get_model_path_by_type(model_type)
            if path:
                paths[model_type] = path
        return paths

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

    def save_models(self, rf_model=None, lstm_model=None, feature_state=None, blend_weight=None):
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

        # Save RF model (no hybrid prefix)
        if rf_model is not None:
            rf_path = save_path / "rf_model.joblib"
            joblib.dump(rf_model, rf_path)
            saved_files['rf_model'] = rf_path

        # Save TabNet model (no hybrid prefix)
        if lstm_model is not None:  # Note: parameter name lstm_model but used for TabNet
            tabnet_path = save_path / "tabnet_model.keras"
            lstm_model.save(tabnet_path)
            saved_files['tabnet_model'] = tabnet_path

        # Save feature engineering state if provided (no hybrid prefix)
        if feature_state is not None:
            feature_path = save_path / "feature_engineer.joblib"
            joblib.dump(feature_state, feature_path)
            saved_files['feature_engineer'] = feature_path

        # Save minimal config
        config_data = {
            'db_type': db_type,
            'created_at': datetime.now().isoformat()
        }

        # Add blend_weight if provided
        if blend_weight is not None:
            config_data['blend_weight'] = blend_weight

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

        # Load RF model (no hybrid prefix)
        rf_path = model_path / "rf_model.joblib"
        if rf_path.exists():
            models['rf_model'] = joblib.load(rf_path)

        # Load TabNet model (no hybrid prefix)
        tabnet_path = model_path / "tabnet_model.keras"
        if tabnet_path.exists():
            models['tabnet_model'] = load_model(tabnet_path)

        # Load feature engineering state (no hybrid prefix)
        feature_path = model_path / "feature_engineer.joblib"
        if feature_path.exists():
            models['feature_state'] = joblib.load(feature_path)

        # Load model config
        config_path = model_path / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                models['model_config'] = json.load(f)

        return models
        
    def load_all_models(self):
        """Load all available model types (legacy, tabnet, feedforward)."""
        all_models = {}
        model_paths = self.get_all_model_paths()
        
        for model_type, model_path in model_paths.items():
            print(f"Loading {model_type} models from: {model_path}")
            
            if model_type == 'legacy':
                # Load legacy RF/LSTM models using existing method
                try:
                    legacy_models = self.load_models(model_path)
                    all_models['legacy'] = legacy_models
                    print(f"  ✅ Loaded legacy models: {list(legacy_models.keys())}")
                except Exception as e:
                    print(f"  ❌ Failed to load legacy models: {e}")
                    
            elif model_type == 'tabnet':
                # Load TabNet model
                try:
                    tabnet_models = self._load_tabnet_models(model_path)
                    all_models['tabnet'] = tabnet_models
                    print(f"  ✅ Loaded TabNet models: {list(tabnet_models.keys())}")
                except Exception as e:
                    print(f"  ❌ Failed to load TabNet models: {e}")
                    
            elif model_type == 'feedforward':
                # Load Feedforward model
                try:
                    ff_models = self._load_feedforward_models(model_path)
                    all_models['feedforward'] = ff_models
                    print(f"  ✅ Loaded Feedforward models: {list(ff_models.keys())}")
                except Exception as e:
                    print(f"  ❌ Failed to load Feedforward models: {e}")
        
        return all_models
    
    def _load_tabnet_models(self, model_path):
        """Load TabNet model components."""
        models = {}
        
        # Load TabNet model
        tabnet_model_path = model_path / "tabnet_model.zip"
        if tabnet_model_path.exists():
            from pytorch_tabnet.tab_model import TabNetRegressor
            tabnet_model = TabNetRegressor()
            tabnet_model.load_model(str(tabnet_model_path))
            models['tabnet_model'] = tabnet_model
            
        # Load TabNet scaler
        tabnet_scaler_path = model_path / "tabnet_scaler.joblib"
        if tabnet_scaler_path.exists():
            models['tabnet_scaler'] = joblib.load(tabnet_scaler_path)
            
        # Load TabNet config
        tabnet_config_path = model_path / "tabnet_config.json"
        if tabnet_config_path.exists():
            with open(tabnet_config_path, 'r') as f:
                models['tabnet_config'] = json.load(f)
                
        return models
    
    def _load_feedforward_models(self, model_path):
        """Load Feedforward model components."""
        models = {}
        
        # Look for Feedforward model files (could be .h5, .keras, .pkl)
        for suffix in ['.h5', '.keras', '.pkl']:
            ff_model_path = model_path / f"feedforward_model{suffix}"
            if ff_model_path.exists():
                if suffix in ['.h5', '.keras']:
                    # TensorFlow/Keras model
                    try:
                        from tensorflow.keras.models import load_model
                        models['feedforward_model'] = load_model(str(ff_model_path))
                    except ImportError:
                        print("TensorFlow not available for loading Feedforward model")
                elif suffix == '.pkl':
                    # Pickled model
                    models['feedforward_model'] = joblib.load(ff_model_path)
                break
                
        # Look for feedforward config
        ff_config_path = model_path / "feedforward_config.json"
        if ff_config_path.exists():
            with open(ff_config_path, 'r') as f:
                models['feedforward_config'] = json.load(f)
                
        return models

    def _update_config(self, model_path):
        """Update config.yaml with latest legacy model."""
        try:
            with open("config.yaml", 'r') as f:
                config_data = yaml.safe_load(f)

            if 'models' not in config_data:
                config_data['models'] = {}

            # Update old format for backward compatibility
            config_data['models']['latest_model'] = model_path
            
            # Update new format
            if 'latest_models' not in config_data['models']:
                config_data['models']['latest_models'] = {}
            config_data['models']['latest_models']['legacy'] = model_path

            with open("config.yaml", 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

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