import os
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

class Baseconfig(BaseModel):
    rootdir: str
    active_db: str
class FeaturesConfig(BaseModel):
    """Features configuration"""
    features_dir: str
    embedding_dim: int
    default_task_type: str

class ModelsConfig(BaseModel):
    """Models configuration"""
    model_dir: str



class Config(BaseModel):
    """Complete application configuration"""
    cache: CacheConfig
    features: FeaturesConfig
    models: ModelsConfig
    databases: List[Dict[str, Any]]
    base: Baseconfig

    # Allow additional fields for custom config values
    class Config:
        extra = "allow"


class MySQLConfig(BaseModel):
    """MySQL connection configuration"""
    host: str
    user: str
    password: str
    dbname: str



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

    def get_mysql_config(self, db_name: str = "mysql") -> MySQLConfig:
        """
        Get MySQL configuration for the specified database name.

        Args:
            db_name: Name of the MySQL database configuration to use

        Returns:
            MySQLConfig object

        Raises:
            ValueError: If the database is not found or is not MySQL
        """
        for db in self._config.databases:
            if db["name"] == db_name:
                if db["type"] != "mysql":
                    raise ValueError(f"Database '{db_name}' is not a MySQL database")

                return MySQLConfig(
                    host=db["host"],
                    user=db["user"],
                    password=db["password"],
                    dbname=db["dbname"]
                )

        raise ValueError(f"MySQL database '{db_name}' not found in configuration")

    def get_active_db_path(self) -> str:
        """
        Retrieve the path of the base.active_db database from the configuration.
        """
        config = self._load_config()
        active_db = config.base.active_db
        if not active_db:
            raise KeyError("'base.active_db' not found in configuration.")

        # Find the database configuration with the matching name
        databases = config.databases
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

    def get_cache_file_path(self, cache_type: str, ensure_dir_exists: bool = True) -> Path:
        """
        Get the file path for a specific cache type based on config.

        Args:
            cache_type: Type of cache
            ensure_dir_exists: Create directory if it doesn't exist

        Returns:
            Path object for the cache file
        """
        cache_dir = self.get_cache_path(cache_type, ensure_dir_exists)

        # Get filename from config if available
        try:
            # Try to get the filename from config.cache.types mapping
            filename = self.config._config.cache.types.get(cache_type)
            if not filename:
                # If not found in the config, use cache_type as the filename
                filename = f"{cache_type}.parquet"
        except (AttributeError, KeyError):
            # Fallback if there's an issue with the config
            filename = f"{cache_type}.parquet"

        return cache_dir / filename

    def get_feature_store_dir(self) -> str:
        """
        Get feature store directory path
        """
        return self._config.features.features_dir

    def get_model_paths(config, model_name: str = 'hybrid') -> Dict[str, Any]:
        """
        Get model paths based on model name.

        Args:
            config: Configuration
            model_name: Name of the model architecture

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

        # Define paths
        model_paths = {
            'model_path': os.path.join(model_dir, model_name),
            'logs': os.path.join(model_dir, model_name, 'logs'),
            'artifacts': {
                'rf_model': f"{model_name}_rf_model.joblib",
                'lstm_model': f"{model_name}_lstm_model",
                'feature_engineer': f"{model_name}_feature_engineer.joblib"
            }
        }

        # Ensure directories exist
        for path in [model_paths['model_path'], model_paths['logs']]:
            os.makedirs(path, exist_ok=True)

        return model_paths

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


# Convenience functions for backward compatibility
def get_sqlite_dbpath(db_name: str = "full") -> str:
    """Get SQLite database path from config"""
    return AppConfig().get_sqlite_dbpath(db_name)


def get_mysql_config(db_name: str = "mysql") -> MySQLConfig:
    """Get MySQL configuration"""
    return AppConfig().get_mysql_config(db_name)


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
    test_config()