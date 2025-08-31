import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from pydantic import BaseModel, Field
import yaml

# Configuration paths
CONFIG_PATH = os.getenv('CONFIG_PATH', 'config.yaml')


class DatabaseEntry(BaseModel):
    """Individual database configuration entry"""
    name: str
    type: str
    description: Optional[str] = None

    # SQLite specific
    path: Optional[str] = None

    # MySQL specific
    host: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    dbname: Optional[str] = None


class CacheConfig(BaseModel):
    """Cache configuration"""
    base_path: str
    types: Dict[str, str]
    use_cache: bool = True

class Baseconfig(BaseModel):
    rootdir: str
    active_db: str

class FeaturesConfig(BaseModel):
    """Features configuration"""
    features_dir: str
    embedding_dim: int
    default_task_type: str
    clean_after_embedding: bool = True
    keep_identifiers: bool = False

class ModelsConfig(BaseModel):
    """Models configuration"""
    model_dir: str

class LSTMConfig(BaseModel):
    """LSTM configuration"""
    sequence_length: int = 5
    step_size: int = 1
    sequential_features: List[str] = []
    static_features: List[str] = []

class DatasetConfig(BaseModel):
    """Dataset splitting configuration"""
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

class MySQLConfig(BaseModel):
    """MySQL connection configuration"""
    host: str
    user: str
    password: str
    dbname: str

class Config(BaseModel):
    """Complete application configuration"""
    cache: CacheConfig
    features: FeaturesConfig
    models: ModelsConfig
    databases: List[Dict[str, Any]]
    base: Baseconfig
    lstm: Optional[LSTMConfig]
    dataset: Optional[DatasetConfig]

    # Allow additional fields for custom config values
    class Config:
        extra = "allow"

class AppConfig:
    """
    Application configuration class that loads settings from config.yaml
    using Pydantic for validation.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AppConfig with optional custom config path
        """
        self.config_path = config_path or CONFIG_PATH
        self._config = self._load_config()

    def _load_config(self) -> Config:
        """
        Load the configuration from yaml file and validate with Pydantic.
        Raises exception if config is missing or invalid.
        """
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                return Config(**config_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}. Please create a valid config.yaml.")
        except Exception as e:
            raise ValueError(f"Error parsing configuration from {self.config_path}: {str(e)}")

    def get_sqlite_dbpath(self, db_name) -> str:
        """
        Get the SQLite database path for the specified database name.

        Args:
            db_name: Name of the database to get path for

        Returns:
            Path to the SQLite database

        Raises:
            ValueError: If the database is not found or is not SQLite
        """
        for db in self._config.databases:
            if db["name"] == db_name:
                if db["type"] != "sqlite":
                    raise ValueError(f"Database '{db_name}' is not a SQLite database")
                if "path" not in db or not db["path"]:
                    raise ValueError(f"Database '{db_name}' does not have a path specified")
                return db["path"]

        raise ValueError(f"Database '{db_name}' not found in configuration")

    def get_mysql_config(self, db_name: str = None) -> MySQLConfig:
        """
        Get MySQL configuration with optional database name override.

        Args:
            db_name: Optional database name to connect to (overrides dbname in config)

        Returns:
            MySQLConfig object with connection parameters

        Raises:
            ValueError: If MySQL database configuration not found
        """
        # Find the MySQL database configuration
        mysql_db = None
        for db in self._config.databases:
            if db["name"] == "mysql" and db["type"] == "mysql":
                mysql_db = db
                break

        if mysql_db is None:
            raise ValueError("MySQL database configuration not found in configuration")

        # Create MySQLConfig, using provided db_name or default from config
        return MySQLConfig(
            host=mysql_db["host"],
            user=mysql_db["user"],
            password=mysql_db["password"],
            dbname=db_name if db_name is not None else mysql_db["dbname"]
        )

    def get_active_db_path(self) -> str:
        """
        Retrieve the path of the base.active_db database from the configuration.
        """
        active_db = self._config.base.active_db
        if not active_db:
            raise KeyError("'base.active_db' not found in configuration.")

        # Find the database configuration with the matching name
        databases = self._config.databases
        db_config = next((db for db in databases if db['name'] == active_db), None)
        if not db_config:
            raise KeyError(f"Database configuration for '{active_db}' not found in the configuration.")

        return db_config['path']

    def get_cache_dir(self):
        """
        Get the base cache directory path.

        Returns:
            Path to the cache directory
        """
        # Access the cache base_path directly from the Pydantic model
        cache_dir = self._config.cache.base_path

        # Ensure it's an absolute path
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), cache_dir)

        return cache_dir

    def get_feature_store_dir(self) -> str:
        """
        Get feature store directory path
        """
        return self._config.features.features_dir

    def get_default_embedding_dim(self) -> int:
        """
        Get default embedding dimension
        """
        return self._config.features.embedding_dim

    def get_default_task_type(self) -> str:
        """
        Get default task type
        """
        return self._config.features.default_task_type

    def list_databases(self) -> List[Dict[str, Any]]:
        """
        Get a list of all configured databases

        Returns:
            List of database configurations as dictionaries
        """
        return self._config.databases

    def get_database_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a database configuration by name

        Args:
            name: Name of the database to get

        Returns:
            Database configuration dict if found, None otherwise
        """
        for db in self._config.databases:
            if db["name"] == name:
                return db
        return None

    def get_dataset_config(self) -> Dict[str, Any]:
        """
        Get dataset splitting configuration

        Returns:
            Dictionary with dataset splitting parameters
        """
        if hasattr(self._config, 'dataset'):
            return {
                'test_size': self._config.dataset.test_size,
                'val_size': self._config.dataset.val_size,
                'random_state': self._config.dataset.random_state
            }

        # Default values
        return {
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42
        }

    def get_features_config(self) -> Dict[str, Any]:
        """
        Get feature processing configuration

        Returns:
            Dictionary with feature processing parameters
        """
        features_config = {
            'embedding_dim': self._config.features.embedding_dim,
            'default_task_type': self._config.features.default_task_type,
            'features_dir': self._config.features.features_dir
        }

        # Add optional parameters if available
        if hasattr(self._config.features, 'clean_after_embedding'):
            features_config['clean_after_embedding'] = self._config.features.clean_after_embedding

        if hasattr(self._config.features, 'keep_identifiers'):
            features_config['keep_identifiers'] = self._config.features.keep_identifiers

        return features_config

    def should_use_cache(self) -> bool:
        """
        Get global cache setting

        Returns:
            Boolean indicating whether to use cache by default
        """
        if hasattr(self._config.cache, 'use_cache'):
            return self._config.cache.use_cache
        return True

    def get_lstm_config(self) -> Dict[str, Any]:
        """
        Get LSTM configuration parameters

        Returns:
            Dictionary with LSTM parameters
        """
        lstm_config = {}

        # Get LSTM parameters from config if available
        if hasattr(self._config, 'lstm'):
            lstm_config = {
                'sequence_length': self._config.lstm.sequence_length,
                'step_size': self._config.lstm.step_size
            }

            # Add feature lists if defined
            if hasattr(self._config.lstm, 'sequential_features') and self._config.lstm.sequential_features:
                lstm_config['sequential_features'] = self._config.lstm.sequential_features

            if hasattr(self._config.lstm, 'static_features') and self._config.lstm.static_features:
                lstm_config['static_features'] = self._config.lstm.static_features
        else:
            # Default values
            lstm_config = {
                'sequence_length': 5,
                'step_size': 1
            }

        return lstm_config

    @staticmethod
    def get_model_paths(config, model_name: str = 'hybrid_model', model_type: str = None) -> Dict[str, Any]:
        """
        Get model paths based on model name and active database.

        Args:
            config: Configuration
            model_name: Name of the model (defaults to 'hybrid_model')
            model_type: Type of model folder ('hybrid_model' or 'incremental_models')

        Returns:
            Dictionary of model paths
        """
        # Get model dir based on config structure
        model_dir = None

        # Try different ways to access model_dir based on config structure
        if isinstance(config, dict):
            if 'models' in config and 'model_dir' in config['models']:
                model_dir = config['models']['model_dir']
        elif hasattr(config, 'models'):
            if hasattr(config.models, 'model_dir'):
                model_dir = config.models.model_dir

        # If model_dir not found, use default
        if not model_dir:
            model_dir = './models'

        # Get active database to determine subdirectory
        active_db = None
        if isinstance(config, dict):
            if 'base' in config and 'active_db' in config['base']:
                active_db = config['base']['active_db']
        elif hasattr(config, 'base'):
            if hasattr(config.base, 'active_db'):
                active_db = config.base.active_db

        # Default to "2years" if not found
        if not active_db or active_db == "full":
            active_db = "2years"

        # Determine model type if not specified
        if model_type is None:
            if model_name == 'hybrid_model':
                model_type = 'hybrid_model'
            else:
                model_type = 'incremental_models'

        # Build complete path with active_db and model type
        complete_model_dir = os.path.join(model_dir, active_db, model_type)

        # Define paths
        model_paths = {
            'model_path': complete_model_dir,
            'logs': os.path.join(complete_model_dir, 'logs'),
            'artifacts': {
                'rf_model': f"hybrid_rf_model.joblib",
                'lstm_model': f"hybrid_lstm_model",
                'feature_engineer': f"hybrid_feature_engineer.joblib"
            },
            'active_db': active_db,
            'model_type': model_type
        }

        # Ensure directories exist
        for path in [model_paths['model_path'], model_paths['logs']]:
            os.makedirs(path, exist_ok=True)

        return model_paths


# Convenience functions for backward compatibility
def get_sqlite_dbpath(db_name: str = "full") -> str:
    """Get SQLite database path from config"""
    return AppConfig().get_sqlite_dbpath(db_name)


def get_mysql_config(db_name: str = None) -> MySQLConfig:
    """Get MySQL configuration with optional db name override"""
    return AppConfig().get_mysql_config(db_name)


# Environment detection functions
def detect_environment():
    """
    Detect if we're running on vast.ai or localhost.

    Returns:
        str: 'vast.ai' or 'localhost'
    """
    return 'vast.ai' if os.getenv('VAST_CONTAINERLABEL') else 'localhost'


def setup_pythonpath():
    """
    Automatically set PYTHONPATH based on detected environment.
    """
    env_type = detect_environment()
    print(f"🔍 Detected environment: {env_type}")

    if env_type == 'vast.ai':
        # Vast.ai setup - assume workspace is mounted at /workspace
        project_root = Path('/workspace')
    else:
        # Localhost setup - use config or current directory
        try:
            app_config = AppConfig()
            project_root = Path(app_config._config.base.rootdir)
        except Exception as e:
            print(f"⚠️ Could not load config ({e}), using current directory")
            project_root = Path.cwd()

    # Define paths to add
    paths_to_add = [
        str(project_root),
        str(project_root / 'core'),
        str(project_root / 'utils'),
        str(project_root / 'model_training'),
    ]

    # Add paths to sys.path if not already present
    added_paths = []
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            added_paths.append(path)

    # Also set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_paths = [p for p in paths_to_add if p not in current_pythonpath.split(os.pathsep)]

    if new_paths:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = os.pathsep.join(new_paths + [current_pythonpath])
        else:
            os.environ['PYTHONPATH'] = os.pathsep.join(new_paths)

    print(f"✅ Added {len(added_paths)} paths to sys.path")
    for path in added_paths:
        print(f"   📁 {path}")

    return env_type, project_root


def init_environment():
    """
    Initialize environment and return config.
    Call this at the top of your scripts.
    """
    env_type, project_root = setup_pythonpath()

    try:
        app_config = AppConfig()
        print(f"✅ Configuration loaded successfully")
        return app_config, env_type
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return None, env_type


# Direct testing function
def test_config():
    """Test the configuration loading and access"""
    try:
        config = AppConfig()
        print("Config validation test:")
        print(f"Cache dir: {config.get_cache_dir()}")
        print(f"Feature store dir: {config.get_feature_store_dir()}")
        print(f"Embedding dimension: {config.get_default_embedding_dim()}")

        # Test database configurations
        print("\nConfigured databases:")
        for db in config.list_databases():
            print(f"  - {db['name']} ({db['type']}): {db.get('description', 'No description')}")

        # Test SQLite path
        try:
            sqlite_path = config.get_sqlite_dbpath("full")
            print(f"\nFull SQLite database path: {sqlite_path}")
        except ValueError as e:
            print(f"\nSQLite database error: {str(e)}")

        # Test MySQL config if available
        try:
            mysql = config.get_mysql_config()
            print(f"MySQL config: host={mysql.host}, user={mysql.user}, dbname={mysql.dbname}")
        except ValueError as e:
            print(f"MySQL database error: {str(e)}")

        print("\nConfiguration valid and accessible.")
    except Exception as e:
        print(f"Configuration error: {str(e)}")
        print("Please ensure your config.yaml is properly configured.")


if __name__ == "__main__":
    # Test environment detection and config loading
    env_type, project_root = setup_pythonpath()
    print(f"\n📊 Environment Summary:")
    print(f"   Environment: {env_type}")
    print(f"   Project Root: {project_root}")
    print(f"   VAST_CONTAINERLABEL: {os.getenv('VAST_CONTAINERLABEL', 'Not set')}")

    # Test config loading
    app_config, _ = init_environment()
    if app_config:
        print(f"   Active DB: {app_config._config.base.active_db}")
    
    # Run the original test
    test_config()