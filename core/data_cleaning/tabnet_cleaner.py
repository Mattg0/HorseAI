"""
Comprehensive data cleaning pipeline for TabNet training.
Fixes NaN issues while preserving racing domain semantics and data integrity.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Optional
from core.calculators.musique_calculation import MusiqueFeatureExtractor


class TabNetDataCleaner:
    """
    Comprehensive data cleaner for TabNet training that preserves racing domain meaning.
    """
    
    def __init__(self):
        """Initialize cleaner with racing domain knowledge."""
        self.musique_extractor = MusiqueFeatureExtractor()
        
        # Racing position mappings (reuse existing domain logic)
        self.racing_position_map = {
            'DAI': 15.0,     # Distant finish
            'DISTANTE': 15.0,
            'DIST': 15.0,
            'ARR': 16.0,     # Did not finish
            'ARRETE': 16.0,
            'STOPPED': 16.0,
            'TOMBE': 17.0,   # Fell
            'FALLEN': 17.0,
            'NP': 18.0,      # Did not start
            'NON_PARTANT': 18.0,
            'DID_NOT_START': 18.0
        }
        
        # Empty string fields (Category 1)
        self.empty_string_fields = [
            'derniereplace', 'tempstot', 'oeil', 'defoeil', 'defoeilPrec',
            'proprietaire', 'dernierOeil'
        ]
        
        # High null fields (Category 3)
        self.high_null_fields = [
            'recordG', 'proprietaire'
        ]
        
        # Mixed type fields (Category 2)
        self.mixed_type_fields = [
            'derniereplace', 'dernierOeil', 'dernierEnt'
        ]
        
        # Statistics for logging
        self.cleaning_stats = {
            'empty_strings_replaced': 0,
            'racing_positions_converted': 0,
            'time_records_parsed': 0,
            'trainer_fields_cleaned': 0,
            'final_nans_filled': 0,
            'total_nans_before': 0,
            'total_nans_after': 0
        }
    
    def comprehensive_data_cleaning(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Clean data for TabNet training while preserving racing domain meaning.
        
        Args:
            df: Input DataFrame to clean
            verbose: Whether to log cleaning statistics
            
        Returns:
            Cleaned DataFrame ready for TabNet training
        """
        if verbose:
            print("[DATA-CLEAN] Starting comprehensive data cleaning for TabNet...")
        
        # Track initial state
        df_clean = df.copy()
        initial_nan_count = df_clean.isnull().sum().sum()
        self.cleaning_stats['total_nans_before'] = initial_nan_count
        
        if verbose:
            print(f"[DATA-CLEAN] Initial NaN count: {initial_nan_count}")
        
        # Category 1: Empty string replacement
        df_clean = self._clean_empty_strings(df_clean, verbose)
        
        # Category 2: Mixed data type fields
        df_clean = self._clean_mixed_type_fields(df_clean, verbose)
        
        # Category 3: Parse specialized formats
        df_clean = self._parse_specialized_formats(df_clean, verbose)
        
        # Category 4: Handle high null fields
        df_clean = self._handle_high_null_fields(df_clean, verbose)
        
        # Category 5: Final NaN elimination
        df_clean = self._final_nan_elimination(df_clean, verbose)
        
        # Track final state
        final_nan_count = df_clean.isnull().sum().sum()
        self.cleaning_stats['total_nans_after'] = final_nan_count
        
        if verbose:
            self._log_cleaning_summary()
        
        return df_clean
    
    def _clean_empty_strings(self, df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        """Clean Category 1: Empty string issues."""
        empty_strings_replaced = 0
        
        for field in self.empty_string_fields:
            if field in df.columns:
                # Replace empty strings with NaN
                mask = (df[field] == '') | (df[field] == ' ')
                count = mask.sum()
                if count > 0:
                    df[field] = df[field].replace(['', ' '], np.nan).infer_objects(copy=False)
                    empty_strings_replaced += count
                    if verbose:
                        print(f"[DATA-CLEAN] {field}: Replaced {count} empty strings with NaN")
        
        self.cleaning_stats['empty_strings_replaced'] = empty_strings_replaced
        return df
    
    def _clean_mixed_type_fields(self, df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        """Clean Category 2: Mixed data type issues."""
        racing_positions_converted = 0
        trainer_fields_cleaned = 0
        
        # Handle derniereplace with racing position codes
        if 'derniereplace' in df.columns:
            converted_count = 0
            for code, numeric_value in self.racing_position_map.items():
                mask = df['derniereplace'].astype(str).str.upper() == code
                count = mask.sum()
                if count > 0:
                    df.loc[mask, 'derniereplace'] = numeric_value
                    converted_count += count
            
            racing_positions_converted += converted_count
            if verbose and converted_count > 0:
                print(f"[DATA-CLEAN] derniereplace: Converted {converted_count} racing codes to numeric")
        
        # Handle dernierEnt trainer field (keep trainer names as categorical indicators)
        if 'dernierEnt' in df.columns:
            # Convert trainer names to categorical codes, but preserve '0' as numeric
            trainer_mask = df['dernierEnt'].astype(str).str.contains(r'[a-zA-Z]', regex=True, na=False)
            trainer_count = trainer_mask.sum()
            if trainer_count > 0:
                # For now, convert trainer names to 0 to indicate "trainer change"
                # This preserves the semantic meaning while making it numeric
                df.loc[trainer_mask, 'dernierEnt'] = 0
                trainer_fields_cleaned += trainer_count
                if verbose:
                    print(f"[DATA-CLEAN] dernierEnt: Cleaned {trainer_count} trainer name entries")
        
        self.cleaning_stats['racing_positions_converted'] = racing_positions_converted
        self.cleaning_stats['trainer_fields_cleaned'] = trainer_fields_cleaned
        return df
    
    def _parse_specialized_formats(self, df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        """Clean Category 4: Specialized format fields."""
        time_records_parsed = 0
        
        # Handle recordG time format: "1'29"4" -> milliseconds
        if 'recordG' in df.columns:
            original_count = df['recordG'].notna().sum()
            
            def parse_recordG_to_milliseconds(time_str):
                """Convert "1'29"4" format to milliseconds."""
                if pd.isna(time_str) or str(time_str).strip() == '':
                    return np.nan
                
                try:
                    time_str = str(time_str).strip()
                    
                    # Pattern: minute'second"decisecond
                    pattern = r"(\d+)'(\d+)\"(\d+)"
                    match = re.match(pattern, time_str)
                    
                    if match:
                        minutes = int(match.group(1))
                        seconds = int(match.group(2))
                        deciseconds = int(match.group(3))
                        
                        # Convert to milliseconds
                        total_ms = (minutes * 60 * 1000) + (seconds * 1000) + (deciseconds * 100)
                        return float(total_ms)
                    else:
                        # Invalid format
                        return np.nan
                        
                except (ValueError, AttributeError):
                    return np.nan
            
            df['recordG'] = df['recordG'].apply(parse_recordG_to_milliseconds)
            parsed_count = df['recordG'].notna().sum()
            time_records_parsed = parsed_count
            
            if verbose:
                print(f"[DATA-CLEAN] recordG: Parsed {parsed_count}/{original_count} time records to milliseconds")
        
        self.cleaning_stats['time_records_parsed'] = time_records_parsed
        return df
    
    def _handle_high_null_fields(self, df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        """Clean Category 3: High null percentage fields."""
        
        for field in self.high_null_fields:
            if field in df.columns:
                null_pct = (df[field].isnull().sum() / len(df)) * 100
                if null_pct > 50:
                    if verbose:
                        print(f"[DATA-CLEAN] {field}: {null_pct:.1f}% null - will use domain default")
        
        return df
    
    def _final_nan_elimination(self, df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        """Category 5: Final NaN elimination with domain-appropriate defaults."""
        final_nans_filled = 0
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Fill numeric columns with 0 (racing default for missing data)
        for col in numeric_columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                df[col] = df[col].fillna(0.0)
                final_nans_filled += nan_count
                if verbose and nan_count > 0:
                    print(f"[DATA-CLEAN] {col}: Filled {nan_count} NaN with 0.0")
        
        # Fill object columns with appropriate defaults
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                # Use empty string for text fields in racing context
                df[col] = df[col].fillna('')
                final_nans_filled += nan_count
                if verbose and nan_count > 0:
                    print(f"[DATA-CLEAN] {col}: Filled {nan_count} NaN with empty string")
        
        self.cleaning_stats['final_nans_filled'] = final_nans_filled
        return df
    
    def _log_cleaning_summary(self):
        """Log comprehensive cleaning summary."""
        print(f"\n[DATA-CLEAN] === CLEANING SUMMARY ===")
        print(f"[DATA-CLEAN] Empty strings replaced: {self.cleaning_stats['empty_strings_replaced']}")
        print(f"[DATA-CLEAN] Racing positions converted: {self.cleaning_stats['racing_positions_converted']}")
        print(f"[DATA-CLEAN] Time records parsed: {self.cleaning_stats['time_records_parsed']}")
        print(f"[DATA-CLEAN] Trainer fields cleaned: {self.cleaning_stats['trainer_fields_cleaned']}")
        print(f"[DATA-CLEAN] Final NaNs filled: {self.cleaning_stats['final_nans_filled']}")
        print(f"[DATA-CLEAN] Total NaN count: {self.cleaning_stats['total_nans_before']} → {self.cleaning_stats['total_nans_after']}")
        
        improvement = self.cleaning_stats['total_nans_before'] - self.cleaning_stats['total_nans_after']
        print(f"[DATA-CLEAN] Improvement: -{improvement} NaN values ({improvement} cleaned)")
        print(f"[DATA-CLEAN] === CLEANING COMPLETE ===\n")
    
    def validate_tabnet_compatibility(self, df: pd.DataFrame, verbose: bool = True) -> bool:
        """
        Validate that DataFrame is compatible with TabNet training.
        
        Args:
            df: DataFrame to validate
            verbose: Whether to log validation details
            
        Returns:
            True if compatible, False otherwise
        """
        issues = []
        
        # Check for NaN values
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            issues.append(f"Contains {nan_count} NaN values")
        
        # Check for infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = 0
        for col in numeric_cols:
            inf_count += np.isinf(df[col]).sum()
        
        if inf_count > 0:
            issues.append(f"Contains {inf_count} infinite values")
        
        # Check data types
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0 and verbose:
            print(f"[VALIDATION] Warning: {len(object_cols)} object columns remain")
        
        if issues:
            if verbose:
                print(f"[VALIDATION] ❌ TabNet compatibility issues found:")
                for issue in issues:
                    print(f"[VALIDATION]   - {issue}")
            return False
        else:
            if verbose:
                print(f"[VALIDATION] ✅ DataFrame is TabNet compatible")
                print(f"[VALIDATION]   Shape: {df.shape}")
                print(f"[VALIDATION]   Numeric columns: {len(numeric_cols)}")
                print(f"[VALIDATION]   Object columns: {len(object_cols)}")
            return True