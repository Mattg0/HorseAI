# utils/cache_manager.py
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Union
import pandas as pd
import fastparquet as fp

from utils.env_setup import AppConfig


class CacheManager:
    """
    Manages cache locations for different parts of the application.
    Provides paths based on cache type and handles parquet file operations.
    """

    def __init__(self):
        self.config = AppConfig()

        self._ensure_base_dir()

    def _ensure_base_dir(self):
        """Ensure the base cache directory exists."""
        os.makedirs(self.config.get_cache_dir(), exist_ok=True)

    def get_cache_path(self, cache_type: str, ensure_exists: bool = True) -> Path:
        """
        Get the path for a specific cache type.

        Args:
            cache_type: Type of cache (must be defined in config)
            ensure_exists: Create directory if it doesn't exist

        Returns:
            Path object for the cache directory
        """
        if cache_type not in self.config.types:
            raise ValueError(f"Unknown cache type: {cache_type}. Available types: {list(self.config.types.keys())}")

        cache_path = Path(self.config.get_cache_file_path()) / self.config.types[cache_type]

        if ensure_exists:
            os.makedirs(cache_path, exist_ok=True)

        return cache_path

    def get_cache_file_path(self, cache_type: str, filename: str, ensure_dir_exists: bool = True) -> Path:
        """
        Get a path for a specific cache file.

        Args:
            cache_type: Type of cache
            filename: Name of the cache file
            ensure_dir_exists: Create directory if it doesn't exist

        Returns:
            Path object for the cache file
        """
        cache_dir = self.get_cache_path(cache_type, ensure_dir_exists)

        # Add .parquet extension if not present and not another extension
        if not filename.endswith('.parquet') and '.' not in filename:
            filename = f"{filename}.parquet"

        return cache_dir / filename

    def save_dataframe(self, df: pd.DataFrame, cache_type: str, filename: str,
                       compression: str = 'SNAPPY', **kwargs) -> Path:
        """
        Save a pandas DataFrame to parquet format in the cache.

        Args:
            df: DataFrame to save
            cache_type: Type of cache
            filename: Name to save the file as
            compression: Compression algorithm (default: SNAPPY)
            **kwargs: Additional arguments for fastparquet.write

        Returns:
            Path where the file was saved
        """
        file_path = self.get_cache_file_path(cache_type)

        # Save DataFrame using fastparquet
        fp.write(file_path, df, compression=compression, **kwargs)

        return file_path

    def load_dataframe(self, cache_type: str, filename: str) -> Optional[pd.DataFrame]:
        """
        Load a pandas DataFrame from parquet format in the cache.

        Args:
            cache_type: Type of cache
            filename: Name of the file to load

        Returns:
            DataFrame if file exists, None otherwise
        """
        file_path = self.get_cache_file_path(cache_type, filename, ensure_dir_exists=False)

        if not file_path.exists():
            return None

        # Load DataFrame using fastparquet
        return fp.ParquetFile(file_path).to_pandas()

    def file_exists(self, cache_type: str, filename: str) -> bool:
        """
        Check if a file exists in the cache.

        Args:
            cache_type: Type of cache
            filename: Name of the file

        Returns:
            True if file exists, False otherwise
        """
        file_path = self.get_cache_file_path(cache_type, filename, ensure_dir_exists=False)
        return file_path.exists()

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache files.

        Args:
            cache_type: Type of cache to clear, or None to clear all caches
        """
        if cache_type is None:
            # Clear all caches
            for type_name in self.config.types:
                cache_path = self.get_cache_path(type_name, ensure_exists=False)
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                    os.makedirs(cache_path, exist_ok=True)
        else:
            # Clear specific cache
            cache_path = self.get_cache_path(cache_type, ensure_exists=False)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                os.makedirs(cache_path, exist_ok=True)

    def list_files(self, cache_type: str) -> list:
        """
        List all files in a cache directory.

        Args:
            cache_type: Type of cache

        Returns:
            List of filenames
        """
        cache_path = self.get_cache_path(cache_type, ensure_exists=False)

        if not cache_path.exists():
            return []

        return [f.name for f in cache_path.iterdir() if f.is_file()]


# Singleton instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get the singleton cache manager instance."""
    global _cache_manager

    if _cache_manager is None:
        config = load_config()
        _cache_manager = CacheManager(config.cache)

    return _cache_manager


def get_cache_path(cache_type: str, filename: Optional[str] = None) -> Path:
    """
    Helper function to get a cache path.

    Args:
        cache_type: Type of cache
        filename: Optional filename within the cache directory

    Returns:
        Path to cache directory or file
    """
    manager = get_cache_manager()

    if filename:
        return manager.get_cache_file_path(cache_type, filename)
    else:
        return manager.get_cache_path(cache_type)


def save_df_cache(df: pd.DataFrame, cache_type: str, filename: str, **kwargs) -> Path:
    """Helper function to save a DataFrame to cache."""
    return get_cache_manager().save_dataframe(df, cache_type, filename, **kwargs)


def load_df_cache(cache_type: str, filename: str) -> Optional[pd.DataFrame]:
    """Helper function to load a DataFrame from cache."""
    return get_cache_manager().load_dataframe(cache_type, filename)