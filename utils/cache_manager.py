import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Union, Any
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
        cache_path = Path(self.config.get_cache_dir()) / cache_type

        if ensure_exists:
            os.makedirs(cache_path, exist_ok=True)

        return cache_path

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

    def save_dataframe(self, df: pd.DataFrame, cache_type: str, *args,
                       compression: str = 'SNAPPY', **kwargs) -> Path:
        """
        Save a pandas DataFrame to parquet format in the cache.

        This method accepts additional positional and keyword arguments which are ignored,
        to maintain compatibility with existing code that might pass extra arguments.

        Args:
            df: DataFrame to save
            cache_type: Type of cache (used as both directory and filename)
            compression: Compression algorithm (default: SNAPPY)
            **kwargs: Additional arguments for fastparquet.write

        Returns:
            Path where the file was saved
        """
        file_path = self.get_cache_file_path(cache_type)

        # Create a copy to avoid modifying the original
        df_to_save = df.copy()

        # Handle problematic columns
        problematic_columns = ['reunion', 'reun']  # Add other columns as needed
        for col in problematic_columns:
            if col in df_to_save.columns:
                # Convert to string to avoid type conversion issues
                df_to_save[col] = df_to_save[col].astype(str)

        # Save DataFrame using fastparquet
        try:
            fp.write(file_path, df_to_save, compression=compression, **kwargs)
        except Exception as e:
            print(f"Warning: Error saving to cache: {str(e)}. Using original dataframe.")
            # If conversion fails, try with original dataframe
            fp.write(file_path, df, compression=compression, **kwargs)

        return file_path

    def load_dataframe(self, cache_type: str, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        Load a pandas DataFrame from parquet format in the cache.

        This method accepts additional positional and keyword arguments which are ignored,
        to maintain compatibility with existing code that might pass extra arguments.

        Args:
            cache_type: Type of cache (used as both directory and filename)

        Returns:
            DataFrame if file exists, None otherwise
        """
        file_path = self.get_cache_file_path(cache_type, ensure_dir_exists=False)

        if not file_path.exists():
            return None

        # Load DataFrame using fastparquet
        return fp.ParquetFile(file_path).to_pandas()

    def file_exists(self, cache_type: str) -> bool:
        """
        Check if a cache file exists.

        Args:
            cache_type: Type of cache (used as both directory and filename)

        Returns:
            True if file exists, False otherwise
        """
        file_path = self.get_cache_file_path(cache_type, ensure_dir_exists=False)
        return file_path.exists()

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache files.

        Args:
            cache_type: Type of cache to clear, or None to clear all caches
        """
        if cache_type is None:
            # Clear all caches - this removes the entire cache directory
            cache_dir = Path(self.config.get_cache_dir())
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
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
        _cache_manager = CacheManager()

    return _cache_manager


def get_cache_path(cache_type: str) -> Path:
    """
    Helper function to get a cache path.

    Args:
        cache_type: Type of cache

    Returns:
        Path to cache directory or file
    """
    return get_cache_manager().get_cache_path(cache_type)


def save_df_cache(df: pd.DataFrame, cache_type: str, **kwargs) -> Path:
    """Helper function to save a DataFrame to cache."""
    return get_cache_manager().save_dataframe(df, cache_type, **kwargs)


def load_df_cache(cache_type: str) -> Optional[pd.DataFrame]:
    """Helper function to load a DataFrame from cache."""
    return get_cache_manager().load_dataframe(cache_type)