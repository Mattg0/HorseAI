"""
Intelligent earnings imputation based on racing economics and performance indicators.

This module handles missing gainsCarriere (career earnings) data by estimating
earnings from available performance indicators rather than using arbitrary values.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


class EarningsImputer:
    """
    Impute missing career earnings based on racing economics and performance data.

    Strategy hierarchy:
    1. Estimate from win/place record (most reliable)
    2. Use field median for experienced horses (contextual)
    3. Zero for inexperienced horses (acknowledge uncertainty)
    """

    # Average prize money by race type (in euros)
    # Based on French racing economics
    AVERAGE_WIN_PRIZE = {
        'Plat': 50000,      # Flat races
        'Haies': 40000,     # Hurdle races
        'Steeple': 45000,   # Steeplechase
        'A': 35000,         # Trot attelé (harness)
        'M': 30000,         # Trot monté (mounted)
        'default': 40000    # Generic average
    }

    AVERAGE_PLACE_PRIZE = {
        'Plat': 15000,
        'Haies': 12000,
        'Steeple': 13000,
        'A': 10000,
        'M': 8000,
        'default': 12000
    }

    # Minimum races to be considered "experienced" for field median strategy
    MIN_EXPERIENCED_RACES = 5

    def __init__(self, verbose: bool = False):
        """Initialize the earnings imputer."""
        self.verbose = verbose

    def impute_single_horse(
        self,
        gainsCarriere: Optional[float],
        victoirescheval: Optional[int],
        placescheval: Optional[int],
        coursescheval: Optional[int],
        typec: Optional[str] = None,
        field_median: Optional[float] = None
    ) -> float:
        """
        Impute earnings for a single horse.

        Args:
            gainsCarriere: Career earnings (may be None/NaN)
            victoirescheval: Number of victories
            placescheval: Number of places (2nd-3rd)
            coursescheval: Total number of races
            typec: Race type (Plat, Haies, etc.)
            field_median: Median earnings of experienced horses in field

        Returns:
            Imputed earnings value
        """
        # If earnings are valid and positive, return as-is
        if pd.notna(gainsCarriere) and gainsCarriere > 0:
            return float(gainsCarriere)

        # If earnings are negative (data error), treat as missing
        if pd.notna(gainsCarriere) and gainsCarriere < 0:
            if self.verbose:
                print(f"   Warning: Negative earnings detected ({gainsCarriere:.0f}), treating as missing")
            # Continue to imputation strategies below

        # Ensure numeric types
        victories = int(victoirescheval) if pd.notna(victoirescheval) else 0
        places = int(placescheval) if pd.notna(placescheval) else 0
        races = int(coursescheval) if pd.notna(coursescheval) else 0

        # Strategy 1: Estimate from performance if available
        if victories > 0 or places > 0:
            estimated_earnings = self._estimate_from_performance(
                victories, places, races, typec
            )

            if self.verbose:
                print(f"   Imputed from performance: {estimated_earnings:.0f} "
                      f"(W:{victories}, P:{places}, R:{races})")

            return estimated_earnings

        # Strategy 2: Use field median for experienced horses
        if races >= self.MIN_EXPERIENCED_RACES and field_median is not None:
            if self.verbose:
                print(f"   Imputed from field median: {field_median:.0f} "
                      f"({races} races, no wins/places)")

            return field_median

        # Strategy 3: Zero for inexperienced horses
        if self.verbose:
            print(f"   Imputed as zero: inexperienced horse ({races} races)")

        return 0.0

    def _estimate_from_performance(
        self,
        victories: int,
        places: int,
        races: int,
        typec: Optional[str]
    ) -> float:
        """
        Estimate earnings from win/place record.

        Uses race-type-specific average prize money and adjusts for
        horses with many races (likely also earned smaller prizes).
        """
        # Get race-type specific prize money
        race_type = typec if typec in self.AVERAGE_WIN_PRIZE else 'default'
        avg_win = self.AVERAGE_WIN_PRIZE[race_type]
        avg_place = self.AVERAGE_PLACE_PRIZE[race_type]

        # Base estimate from wins and places
        estimated = (victories * avg_win) + (places * avg_place)

        # Add estimated earnings from other finishes
        # Horses with many races likely earned smaller prizes
        other_races = max(0, races - victories - places)
        if other_races > 0:
            # Estimate ~5k average for 4th-6th finishes (conservative)
            estimated += other_races * 5000

        return float(estimated)

    def impute_race_field(
        self,
        race_df: pd.DataFrame,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Impute missing earnings for all horses in a race.

        This method:
        1. Calculates field median from horses with valid earnings
        2. Imputes missing values using the hierarchy strategy
        3. Preserves original data when available

        Args:
            race_df: DataFrame with race participants
            inplace: Whether to modify DataFrame in place

        Returns:
            DataFrame with imputed gainsCarriere values
        """
        if not inplace:
            race_df = race_df.copy()

        # Ensure required columns exist
        required_cols = ['victoirescheval', 'placescheval', 'coursescheval']
        for col in required_cols:
            if col not in race_df.columns:
                race_df[col] = 0

        # Ensure gainsCarriere column exists
        if 'gainsCarriere' not in race_df.columns:
            race_df['gainsCarriere'] = np.nan

        # Calculate field median from experienced horses with valid earnings
        field_median = self._calculate_field_median(race_df)

        # Get race type (assume same for all horses in race)
        typec = race_df['typec'].iloc[0] if 'typec' in race_df.columns and len(race_df) > 0 else None

        # Count how many need imputation
        missing_count = race_df['gainsCarriere'].isna().sum() + (race_df['gainsCarriere'] <= 0).sum()

        if missing_count > 0 and self.verbose:
            print(f"\n🔧 Imputing earnings for {missing_count}/{len(race_df)} horses")
            if field_median is not None:
                print(f"   Field median: {field_median:.0f} (from experienced horses)")

        # Impute each horse
        for idx in race_df.index:
            race_df.at[idx, 'gainsCarriere'] = self.impute_single_horse(
                gainsCarriere=race_df.at[idx, 'gainsCarriere'],
                victoirescheval=race_df.at[idx, 'victoirescheval'],
                placescheval=race_df.at[idx, 'placescheval'],
                coursescheval=race_df.at[idx, 'coursescheval'],
                typec=typec,
                field_median=field_median
            )

        return race_df

    def _calculate_field_median(self, race_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate median earnings from experienced horses with valid earnings.

        Only considers horses with:
        - Valid (non-null, positive) earnings
        - Minimum number of races (experienced)
        """
        # Filter to experienced horses with valid earnings
        mask = (
            race_df['gainsCarriere'].notna() &
            (race_df['gainsCarriere'] > 0) &
            (race_df['coursescheval'] >= self.MIN_EXPERIENCED_RACES)
        )

        valid_earnings = race_df.loc[mask, 'gainsCarriere']

        if len(valid_earnings) >= 3:  # Need at least 3 horses for reliable median
            return float(valid_earnings.median())

        return None

    def get_imputation_stats(self, race_df: pd.DataFrame) -> Dict[str, any]:
        """
        Get statistics about earnings imputation needs.

        Args:
            race_df: DataFrame with race participants

        Returns:
            Dictionary with imputation statistics
        """
        stats = {
            'total_horses': len(race_df),
            'missing_earnings': 0,
            'zero_earnings': 0,
            'valid_earnings': 0,
            'experienced_horses': 0,
            'horses_with_wins': 0,
            'horses_with_places': 0
        }

        if 'gainsCarriere' not in race_df.columns:
            return stats

        stats['missing_earnings'] = race_df['gainsCarriere'].isna().sum()
        stats['zero_earnings'] = (race_df['gainsCarriere'] == 0).sum()
        stats['valid_earnings'] = (
            race_df['gainsCarriere'].notna() &
            (race_df['gainsCarriere'] > 0)
        ).sum()

        if 'coursescheval' in race_df.columns:
            stats['experienced_horses'] = (
                race_df['coursescheval'] >= self.MIN_EXPERIENCED_RACES
            ).sum()

        if 'victoirescheval' in race_df.columns:
            stats['horses_with_wins'] = (race_df['victoirescheval'] > 0).sum()

        if 'placescheval' in race_df.columns:
            stats['horses_with_places'] = (race_df['placescheval'] > 0).sum()

        return stats


def impute_earnings_safe(
    race_df: pd.DataFrame,
    verbose: bool = False,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Convenience function for earnings imputation with error handling.

    Args:
        race_df: DataFrame with race participants
        verbose: Whether to print imputation details
        inplace: Whether to modify DataFrame in place

    Returns:
        DataFrame with imputed earnings
    """
    try:
        imputer = EarningsImputer(verbose=verbose)
        return imputer.impute_race_field(race_df, inplace=inplace)
    except Exception as e:
        if verbose:
            print(f"Warning: Earnings imputation failed: {e}")
            print(f"   Filling missing values with 0")

        # Fallback: simple zero fill
        if not inplace:
            race_df = race_df.copy()

        if 'gainsCarriere' not in race_df.columns:
            race_df['gainsCarriere'] = 0
        else:
            race_df['gainsCarriere'] = race_df['gainsCarriere'].fillna(0)

        return race_df
