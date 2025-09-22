"""
Alternative models package for horse racing prediction.

This package contains implementations of various machine learning models
that can be used as alternatives to LSTM for horse racing prediction.
"""

from .base_model import BaseModel
from .feedforward_model import FeedforwardModel
from .transformer_model import TransformerModel
from .ensemble_model import EnsembleModel

__all__ = [
    'BaseModel',
    'FeedforwardModel', 
    'TransformerModel',
    'EnsembleModel'
]