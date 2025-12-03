"""
TabNet Feature Selector

Automatic feature selection for TabNet models that:
- Removes constant and sparse features
- Handles correlated features intelligently
- Uses TabNet's own importance for final selection
- Integrates seamlessly into training/prediction pipeline
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict, Any


class TabNetFeatureSelector:
    """
    Automatic feature selection for TabNet models

    Combines sparse feature removal + correlation handling with intelligent
    priority-based selection to create an optimized feature subset.

    Priority Order for Correlated Features:
        1. bytype features (most specific) - KEEP
        2. weighted features (balanced)
        3. global features (least specific) - REMOVE if correlated
        4. non-musique features - ALWAYS KEEP
    """

    def __init__(self,
                 sparse_threshold: float = 0.7,
                 correlation_threshold: float = 0.95,
                 target_features: int = 45):
        """
        Initialize feature selector

        Args:
            sparse_threshold: Max fraction of zeros allowed (default 0.7 = 70%)
            correlation_threshold: Correlation above which to remove features (default 0.95)
            target_features: Target number of features for final selection (default 45)
        """
        self.sparse_threshold = sparse_threshold
        self.correlation_threshold = correlation_threshold
        self.target_features = target_features
        self.selected_features = None
        self.original_features = None  # Store original features for prediction
        self.selection_metadata = {}

    def fit(self, X: pd.DataFrame, feature_importances: Optional[np.ndarray] = None) -> 'TabNetFeatureSelector':
        """
        Select optimal features for TabNet

        Args:
            X: Training features DataFrame
            feature_importances: Optional TabNet feature importances from initial training

        Returns:
            self for chaining
        """
        print(f"\n{'='*70}")
        print("TabNet Feature Selection")
        print(f"{'='*70}")
        print(f"Input features: {len(X.columns)}")

        features_to_keep = list(X.columns)
        removed = {
            'constant': [],
            'sparse': [],
            'correlated': []
        }

        # Step 1: Remove constant features
        print(f"\nStep 1: Removing constant features...")
        for col in X.columns:
            if X[col].nunique() <= 1:
                features_to_keep.remove(col)
                removed['constant'].append(col)

        if removed['constant']:
            print(f"  Removed {len(removed['constant'])} constant features")
        else:
            print(f"  No constant features found")

        # Step 2: Remove sparse features (>threshold zeros)
        print(f"\nStep 2: Removing sparse features (>{self.sparse_threshold*100:.0f}% zeros)...")
        X_remaining = X[features_to_keep]

        for col in features_to_keep.copy():
            zero_pct = (X_remaining[col] == 0).mean()
            if zero_pct > self.sparse_threshold:
                features_to_keep.remove(col)
                removed['sparse'].append(col)

        if removed['sparse']:
            print(f"  Removed {len(removed['sparse'])} sparse features")
            print(f"  Examples: {removed['sparse'][:3]}")
        else:
            print(f"  No sparse features found")

        # Step 3: Handle correlation - keep most specific version
        print(f"\nStep 3: Handling correlated features (>{self.correlation_threshold})...")
        X_remaining = X[features_to_keep]

        if len(X_remaining.columns) > 1:
            corr_matrix = X_remaining.corr().abs()

            # Priority order: bytype > weighted > global
            def get_feature_priority(feature_name: str) -> int:
                """Higher number = higher priority (keep this one)"""
                if 'bytype' in feature_name:
                    return 3  # Most specific - KEEP
                elif 'weighted' in feature_name:
                    return 2  # Balanced
                elif 'global' in feature_name:
                    return 1  # Least specific - REMOVE if correlated
                else:
                    return 4  # Non-musique features - ALWAYS KEEP

            # Find correlated pairs
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            for column in upper.columns:
                if column not in features_to_keep:
                    continue

                correlated = upper[column][upper[column] > self.correlation_threshold].index.tolist()

                if correlated:
                    # Keep feature with higher priority
                    for corr_feature in correlated:
                        if corr_feature in features_to_keep:
                            col_priority = get_feature_priority(column)
                            corr_priority = get_feature_priority(corr_feature)

                            if col_priority < corr_priority:
                                # Remove current column, keep correlated
                                features_to_keep.remove(column)
                                removed['correlated'].append(column)
                                break
                            else:
                                # Remove correlated, keep current
                                features_to_keep.remove(corr_feature)
                                removed['correlated'].append(corr_feature)

        if removed['correlated']:
            print(f"  Removed {len(removed['correlated'])} correlated features")
            print(f"  Examples: {removed['correlated'][:3]}")
        else:
            print(f"  No highly correlated features found")

        # Step 4: If still too many features, use importance ranking
        if feature_importances is not None and len(features_to_keep) > self.target_features:
            print(f"\nStep 4: Selecting top {self.target_features} by importance...")

            # Create importance dict for remaining features
            importance_dict = dict(zip(X.columns, feature_importances))
            remaining_importance = {f: importance_dict.get(f, 0)
                                   for f in features_to_keep}

            # Sort and keep top N
            sorted_features = sorted(remaining_importance.items(),
                                    key=lambda x: x[1],
                                    reverse=True)
            features_to_keep = [f for f, _ in sorted_features[:self.target_features]]

            print(f"  Selected {len(features_to_keep)} features by importance")

        self.selected_features = features_to_keep
        self.original_features = list(X.columns)  # Store original features for prediction
        self.selection_metadata = {
            'original_count': len(X.columns),
            'original_features': list(X.columns),  # Save for prediction
            'selected_count': len(features_to_keep),
            'removed_constant': removed['constant'],
            'removed_sparse': removed['sparse'],
            'removed_correlated': removed['correlated'],
            'sparse_threshold': self.sparse_threshold,
            'correlation_threshold': self.correlation_threshold,
            'target_features': self.target_features
        }

        print(f"\n{'='*70}")
        print(f"FINAL RESULT:")
        print(f"  Input:  {len(X.columns)} features")
        print(f"  Output: {len(features_to_keep)} features")
        print(f"  Reduction: {(1 - len(features_to_keep)/len(X.columns))*100:.1f}%")
        print(f"{'='*70}\n")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from DataFrame

        Args:
            X: Features DataFrame

        Returns:
            DataFrame with selected features only
        """
        if self.selected_features is None:
            raise ValueError("Must call fit() before transform()")

        # Ensure all selected features exist
        missing = set(self.selected_features) - set(X.columns)
        if missing:
            print(f"Warning: {len(missing)} selected features missing in data")
            print(f"  Missing features: {list(missing)[:5]}")
            available_features = [f for f in self.selected_features if f in X.columns]
            return X[available_features]

        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame,
                      feature_importances: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Fit and transform in one step

        Args:
            X: Features DataFrame
            feature_importances: Optional TabNet feature importances

        Returns:
            DataFrame with selected features only
        """
        self.fit(X, feature_importances)
        return self.transform(X)

    def save(self, path: str) -> None:
        """
        Save selected features to file

        Args:
            path: Path to save JSON file
        """
        if self.selected_features is None:
            raise ValueError("Must call fit() before save()")

        save_data = {
            'selected_features': self.selected_features,
            'metadata': self.selection_metadata
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"✓ Saved feature selection to: {path}")

    @classmethod
    def load(cls, path: str) -> 'TabNetFeatureSelector':
        """
        Load selected features from file

        Args:
            path: Path to JSON file

        Returns:
            Loaded TabNetFeatureSelector instance
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Create new instance
        selector = cls()
        selector.selected_features = data['selected_features']
        selector.selection_metadata = data.get('metadata', {})
        # Load original features from metadata for prediction
        selector.original_features = selector.selection_metadata.get('original_features', selector.selected_features)

        print(f"✓ Loaded {len(selector.selected_features)} selected features from {path}")
        return selector

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature selection

        Returns:
            Dictionary with selection summary
        """
        if self.selected_features is None:
            return {'status': 'not_fitted'}

        return {
            'status': 'fitted',
            'selected_count': len(self.selected_features),
            'selected_features': self.selected_features,
            'removed_counts': {
                'constant': len(self.selection_metadata.get('removed_constant', [])),
                'sparse': len(self.selection_metadata.get('removed_sparse', [])),
                'correlated': len(self.selection_metadata.get('removed_correlated', []))
            },
            'parameters': {
                'sparse_threshold': self.sparse_threshold,
                'correlation_threshold': self.correlation_threshold,
                'target_features': self.target_features
            }
        }


def select_tabnet_features(X_train: pd.DataFrame,
                           y_train: Optional[pd.Series] = None,
                           feature_importances: Optional[np.ndarray] = None,
                           sparse_threshold: float = 0.7,
                           correlation_threshold: float = 0.95,
                           target_features: int = 45) -> tuple:
    """
    Convenience function for feature selection

    Args:
        X_train: Training features DataFrame
        y_train: Training target (not used, for compatibility)
        feature_importances: Optional TabNet feature importances
        sparse_threshold: Max fraction of zeros allowed
        correlation_threshold: Correlation threshold for removal
        target_features: Target number of features

    Returns:
        Tuple of (X_selected, selector)
    """
    selector = TabNetFeatureSelector(
        sparse_threshold=sparse_threshold,
        correlation_threshold=correlation_threshold,
        target_features=target_features
    )

    X_selected = selector.fit_transform(X_train, feature_importances)

    return X_selected, selector
