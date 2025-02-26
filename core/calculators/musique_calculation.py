from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import re
import json


class MusiqueFeatureExtractor:
    """Enhanced musique feature extractor with race type analysis."""

    def __init__(self):
        # Race type mapping
        self.race_types = {
            'A': 'Attele',
            'M': 'Monte',
            'P': 'Plat',
            'H': 'Haies',
            'S': 'Steeple',
            'T': 'Trot'
        }

        # DNF markers
        self.dnf_markers = {'D', 'DA', 'DI', 'DP', 'RET', 'NP', 'ABS'}

        # Default statistics
        self.default_stats = {
            'avg_pos': 99.0,
            'recent_perf': 99.0,
            'consistency': 0.0,
            'trend': 0.0,
            'pct_top3': 0.0,
            'nb_courses': 0,
            'dnf_rate': 0.0
        }

    def _parse_performance(self, perf: str) -> Tuple[int, Optional[str]]:
        """
        Parse a single performance entry.

        Args:
            perf: Performance string (e.g., "1A", "2M", "DA")

        Returns:
            Tuple of (position, race_type)
        """
        perf = perf.strip().upper()

        # Check for DNF
        if any(perf.startswith(dnf) for dnf in self.dnf_markers):
            return 99, None

        # Extract number and letter
        match_num = re.search(r'\d+', perf)
        match_type = re.search(r'[AMPHST]', perf)

        position = int(match_num.group()) if match_num else 99
        race_type = match_type.group() if match_type else None

        return position, race_type

    def _calculate_stats(self, positions: List[int]) -> Dict[str, float]:
        """
        Calculate performance statistics from a list of positions.

        Args:
            positions: List of finishing positions

        Returns:
            Dictionary of calculated statistics
        """
        if not positions:
            return self.default_stats.copy()

        recent_positions = positions[:3]

        stats = {
            'avg_pos': float(np.mean(positions)),
            'recent_perf': float(np.mean(recent_positions)) if recent_positions else 99.0,
            'consistency': float(np.std(positions)) if len(positions) > 1 else 0.0,
            'trend': float(positions[0] - np.mean(positions[1:])) if len(positions) >= 2 else 0.0,
            'pct_top3': float(sum(1 for p in positions if p <= 3) / len(positions)),
            'nb_courses': len(positions)
        }

        return stats

    def extract_features(self, musique: str, current_race_type: Optional[str] = None) -> Dict:
        """
        Extract features from musique string with race type analysis.

        Args:
            musique: Performance history string
            current_race_type: Type of the current race (optional)

        Returns:
            Dictionary containing extracted features
        """
        if not musique or pd.isna(musique):
            features = {
                'global': self.default_stats.copy(),
                'by_type': {},
                'weighted': self.default_stats.copy()
            }
            return features

        # Split and parse performances
        performances = musique.split()
        total_races = len(performances)

        # Initialize containers
        all_positions = []
        positions_by_type = {}
        dnf_count = 0

        # Process each performance
        for perf in reversed(performances):  # Most recent first
            position, race_type = self._parse_performance(perf)

            if position == 99:
                dnf_count += 1
            else:
                all_positions.append(position)

                if race_type:
                    if race_type not in positions_by_type:
                        positions_by_type[race_type] = []
                    positions_by_type[race_type].append(position)

        # Calculate statistics
        global_stats = self._calculate_stats(all_positions)
        global_stats['total_races'] = total_races
        global_stats['dnf_rate'] = dnf_count / total_races if total_races > 0 else 0.0

        type_stats = {
            race_type: self._calculate_stats(positions)
            for race_type, positions in positions_by_type.items()
        }

        # Calculate weighted stats if current_race_type is provided
        weighted_stats = self._calculate_weighted_stats(
            global_stats,
            type_stats.get(current_race_type, self.default_stats.copy()) if current_race_type else None,
            current_race_type
        )

        features = {
            'global': global_stats,
            'by_type': type_stats,
            'weighted': weighted_stats
        }

        return features

    def _calculate_weighted_stats(self,
                                  global_stats: Dict[str, float],
                                  type_stats: Optional[Dict[str, float]],
                                  current_race_type: Optional[str]) -> Dict[str, float]:
        """
        Calculate weighted statistics based on global and race type specific stats.

        Args:
            global_stats: Global performance statistics
            type_stats: Race type specific statistics
            current_race_type: Current race type

        Returns:
            Dictionary of weighted statistics
        """
        if not current_race_type or not type_stats:
            return global_stats

        # Weighting factors
        weights = {
            'same_type': 0.7,  # Weight for same race type
            'global': 0.3  # Weight for global stats
        }

        weighted_stats = {}
        for stat in ['avg_pos', 'recent_perf', 'consistency', 'pct_top3']:
            weighted_stats[stat] = (
                    type_stats[stat] * weights['same_type'] +
                    global_stats[stat] * weights['global']
            )

        # Copy non-weighted stats
        weighted_stats['nb_courses'] = type_stats['nb_courses']
        weighted_stats['total_races'] = global_stats['total_races']
        weighted_stats['dnf_rate'] = global_stats['dnf_rate']
        weighted_stats['trend'] = type_stats['trend']  # Use type-specific trend

        return weighted_stats


def main():
    """Example usage."""
    extractor = MusiqueFeatureExtractor()

    # Test cases
    test_cases = [
        ("(23) 3m 7m 3m 5m 3m Dm 5m 6m 4m 3m 5m Da Da", "M" ),  # Mixed with current type A
    ]

    for musique, race_type in test_cases:
        print(f"\nProcessing musique: {musique}")
        print(f"Current race type: {race_type}")

        features = extractor.extract_features(musique, race_type)

        print("\nGlobal stats:")
        print(json.dumps(features['global'], indent=2))

        print("\nStats by type:")
        print(json.dumps(features['by_type'], indent=2))

        print("\nWeighted stats:")
        print(json.dumps(features['weighted'], indent=2))


if __name__ == "__main__":
    main()