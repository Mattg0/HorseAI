import pandas as pd
import numpy as np

class FeatureCleaner:
    def __init__(self):
        # DEPRECATED: These feature lists are no longer used for quinté models
        # The model's feature_columns.json is the authoritative source
        self.constant_features = [
            'is_handicap_quinte', 'handicap_division',
            'weather_clear', 'weather_rain', 'weather_cloudy',
            'post_position_track_bias',
            'che_weighted_nb_courses', 'che_weighted_trend',
            'joc_weighted_nb_courses', 'joc_weighted_trend',
            'ratio_victoires',
            'efficacite_couple', 'regularite_couple', 'progression_couple'
        ]

        self.global_features = [
            'che_global_avg_pos', 'che_global_recent_perf',
            'che_global_consistency', 'che_global_pct_top3',
            'che_global_total_races', 'che_global_dnf_rate',
            'che_global_trend', 'che_global_nb_courses',
            'joc_global_avg_pos', 'joc_global_recent_perf',
            'joc_global_consistency', 'joc_global_pct_top3',
            'joc_global_total_races', 'joc_global_dnf_rate',
            'joc_global_trend', 'joc_global_nb_courses',
            'perf_jockey_hippo'
        ]

    def clean_features(self, X: pd.DataFrame, expected_features: list = None) -> pd.DataFrame:
        """
        Clean features by removing only those NOT in the expected feature list.

        Args:
            X: Feature DataFrame
            expected_features: List of features expected by the model (from feature_columns.json)
                              If None, uses legacy cleanup (for backward compatibility)

        Returns:
            Cleaned DataFrame with only expected features (if provided)
        """
        X_clean = X.copy()

        if expected_features is not None:
            # NEW APPROACH: Keep only features that are in the expected list
            drop_cols = [col for col in X_clean.columns if col not in expected_features]
            X_clean = X_clean.drop(columns=drop_cols, errors='ignore')
            print(f"Feature cleanup: {len(X.columns)} → {len(X_clean.columns)} ({len(drop_cols)} removed, keeping model features)")
        else:
            # LEGACY APPROACH: Remove specific feature groups (used for training)
            drop_cols = []
            for col in self.constant_features:
                if col in X_clean.columns:
                    drop_cols.append(col)

            # REMOVED: bytype feature removal - these are critical model features!
            # bytype_cols = [col for col in X_clean.columns if 'bytype' in col]
            # drop_cols.extend(bytype_cols)

            for col in self.global_features:
                if col in X_clean.columns:
                    drop_cols.append(col)

            X_clean = X_clean.drop(columns=drop_cols, errors='ignore')
            print(f"Feature cleanup: {len(X.columns)} → {len(X_clean.columns)} ({len(drop_cols)} removed)")

        return X_clean

    def apply_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()

        transforms = {
            'recence': 'recence_log',
            'cotedirect': 'cotedirect_log'
        }

        for original, log_version in transforms.items():
            if original in X_transformed.columns:
                X_transformed[log_version] = np.log1p(X_transformed[original])
                X_transformed = X_transformed.drop(columns=[original])

        return X_transformed
