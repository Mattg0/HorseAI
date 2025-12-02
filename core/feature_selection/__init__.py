"""
Feature selection module for model training
"""

from .tabnet_feature_selector import TabNetFeatureSelector, select_tabnet_features

__all__ = ['TabNetFeatureSelector', 'select_tabnet_features']
